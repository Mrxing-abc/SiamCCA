class EnhancedEMA(nn.Module):
    def __init__(self, channels, factor=8, dilation_rate=2):
        super(EnhancedEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        # 原始 1x1 卷积，用于基本特征抽取
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

        # 改进：使用扩展卷积代替原始的 3x3 卷积
        self.conv_dilated = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                      padding=dilation_rate, dilation=dilation_rate)

        # 新增的全局上下文卷积分支
        self.global_context_conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb_feature, t_feature):
        channel_num = rgb_feature.shape[1]
        x = torch.cat((rgb_feature, t_feature), 1)
        b, c, h, w = x.size()

        # 1. 生成全局上下文特征
        global_context = self.global_context_conv(self.agp(x))  # b, c, 1, 1

        # 2. 分组特征计算
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)

        # 使用 1x1 卷积进行基础特征抽取
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        # 基于 1x1 和 Sigmoid 加权的特征
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        # 改进：使用扩展卷积替代原始 3x3 卷积
        x2 = self.conv_dilated(group_x)

        # 生成加权注意力
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        # 结合全局上下文特征，进行加权融合
        final_output = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        out = final_output + global_context.expand_as(final_output)  # 加入全局上下文特征
        return out[:, :channel_num, :, :], out[:, channel_num:, :, :]


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GSAU(nn.Module):
    def __init__(self, n_feats, drop=0.0, k=2, squeeze_factor=15, attn='GLKA'):
        super().__init__()
        i_feats = n_feats * 2
        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)
        return x * self.scale + shortcut


class AMSCM(nn.Module):
    def __init__(self, n_feats, scales=(3, 5, 7)):
        super().__init__()
        self.n_feats = n_feats
        self.scales = scales
       self.norm = LayerNorm(n_feats, data_format='channels_first')

        # Dynamic scale convolutions
        self.convs = nn.ModuleList([
            nn.Conv2d(n_feats, n_feats, scale, 1, scale // 2, groups=n_feats) for scale in self.scales
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))

    def forward(self, x):
        shortcut = x.clone()
        x = self.conv1(x)
        scale_outs = []

        # Apply convolutions with different scales
        for i, conv in enumerate(self.convs):
            scale_outs.append(conv(x))

        # Calculate weighted sum of different scale outputs
        weighted_sum = sum(w * out for w, out in zip(self.scale_weights, scale_outs))

        return weighted_sum + shortcut


class MAB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LFE = GSAU(n_feats)
        self.AMSCM = AMSCM(n_feats)  # Add the new AMSCM module

    def forward(self, x):
        # Apply adaptive multi-scale convolutions
        x = self.AMSCM(x)
	# Extract local features with GSAU
        x = self.LFE(x)
        return x


