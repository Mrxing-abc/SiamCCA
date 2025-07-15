import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch import nn
import cv2
from SiamCCA.config import config
import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image


from SiamCCA.attention import EnhancedEMA, MAB


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns


# 定义生成热力图的函数
def generate_grad_cam_heatmap(model, image, layer_name, use_attention=False):
    model.eval()

    # 选择目标层并设置 Grad-CAM
    grad_cam = pytorch_grad_cam.GradCAMPlusPlus(model=model, target_layers=[layer_name], use_cuda=True)

    # 获取输入图像的梯度和特征图
    grayscale_cam = grad_cam(input_tensor=image, target_category=None)

    # 显示原图像和热力图叠加效果
    grayscale_cam = grayscale_cam[0, :]  # (H, W)

    # 显示热力图与原图叠加
    cam_image = show_cam_on_image(image[0].cpu().numpy().transpose(1, 2, 0), grayscale_cam)
    return cam_image

class SiamCCANet(nn.Module):
    def __init__(self, ):
        super(SiamCCANet, self).__init__()

        self.anchor_num = config.anchor_num
        self.input_size = config.instance_size
        self.score_displacement = int((self.input_size - config.exemplar_size) / config.total_stride)

        self.former_3_layers_featureExtract = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=2),  # 0） stride=2
            nn.BatchNorm2d(96),  # 1）
            nn.MaxPool2d(3, stride=2),  # 2） stride=2
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, 5),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2),  # 6） stride=2
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, 3),
            nn.BatchNorm2d(384),  # 9
            nn.ReLU(inplace=True),
        )

        self.rgb_featureExtract = nn.Sequential(
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )

        self.t_featureExtract = nn.Sequential(
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )

        self.rgb_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.rgb_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.rgb_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

        self.t_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.t_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.t_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.t_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.t_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

        self.attn_rgb_featureExtract = nn.Sequential(
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )

        self.attn_t_featureExtract = nn.Sequential(
            nn.Conv2d(384, 384, 3),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3),
            nn.BatchNorm2d(256),  # 15
        )


        self.attn_rgb_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_rgb_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)

        self.attn_t_conv_cls1 = nn.Conv2d(256, 256 * 2 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_cls2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_r1 = nn.Conv2d(256, 256 * 4 * self.anchor_num, kernel_size=3, stride=1, padding=0)
        self.attn_t_conv_r2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.attn_t_regress_adjust = nn.Conv2d(4 * self.anchor_num, 4 * self.anchor_num, 1)


        self.template_attention_block = MAB(256)

        self.template_attention_block1 = EnhancedEMA(512)

        self.detection_attention_block = MAB(256)

        self.detection_attention_block1 = EnhancedEMA(512)

    def forward(self, rgb_template, rgb_detection, t_template, t_detection):
        N = rgb_template.size(0)
        rgb_template = self.former_3_layers_featureExtract(rgb_template)
        #rgb_template = self.template_attention_block(rgb_template)
        rgb_detection = self.former_3_layers_featureExtract(rgb_detection)
        #rgb_detection = self.detection_attention_block(rgb_detection)
        t_template = self.former_3_layers_featureExtract(t_template)
        #t_template = self.template_attention_block(t_template)
        t_detection = self.former_3_layers_featureExtract(t_detection)
        #t_detection = self.detection_attention_block(t_detection)


        rgb_template_feature = self.rgb_featureExtract(rgb_template)  # [bs,256,6,6]
        #rgb_template_feature = self.template_attention_block(rgb_template_feature)
        rgb_detection_feature = self.rgb_featureExtract(rgb_detection)  # [bs,256,24,24]
        #rgb_detection_feature = self.detection_attention_block(rgb_detection_feature)



        attn_rgb_template_feature = self.attn_rgb_featureExtract(rgb_template)
        #attn_rgb_template_feature = self.template_attention_block(attn_rgb_template_feature)
        # print(attn_rgb_template_feature.shape)
        attn_rgb_detection_feature = self.attn_rgb_featureExtract(rgb_detection)
        #attn_rgb_detection_feature = self.detection_attention_block(attn_rgb_detection_feature)
        # print(attn_rgb_detection_feature.shape)

        t_template_feature = self.t_featureExtract(t_template)  # [bs,256,6,6]
        #t_template_feature = self.template_attention_block(t_template_feature)
        t_detection_feature = self.t_featureExtract(t_detection)  # [bs,256,24,24]
        #t_detection_feature = self.detection_attention_block(t_detection_feature)
        attn_t_template_feature = self.attn_t_featureExtract(t_template)  # [bs,256,6,6]
        #attn_t_template_feature = self.template_attention_block(attn_t_template_feature)
        # print(attn_t_template_feature.shape)
        attn_t_detection_feature = self.attn_t_featureExtract(t_detection)
        #attn_t_detection_feature = self.detection_attention_block(attn_t_detection_feature)
        # print(attn_t_detection_feature.shape)

        attn_rgb_template_feature = self.template_attention_block(attn_rgb_template_feature)
        attn_rgb_detection_feature = self.detection_attention_block(attn_rgb_detection_feature)
        attn_t_template_feature = self.template_attention_block(attn_t_template_feature)
        attn_t_detection_feature = self.detection_attention_block(attn_t_detection_feature)

        #attn_rgb_template_feature, attn_t_template_feature = self.template_attention_block(attn_rgb_template_feature, attn_t_template_feature)
        attn_rgb_template_feature, attn_t_template_feature = self.template_attention_block1(attn_rgb_template_feature,attn_t_template_feature)
        #union = torch.cat((attn_rgb_detection_feature, attn_t_detection_feature), 1)
        #attn_rgb_detection_feature, attn_t_detection_feature = self.detection_attention_block(attn_rgb_detection_feature, attn_t_detection_feature)
        attn_rgb_detection_feature, attn_t_detection_feature = self.detection_attention_block1(attn_rgb_detection_feature, attn_t_detection_feature)
        #union = self.detection_attention_block(union)
        #attn_rgb_detection_feature, attn_t_detection_feature = union[:, :256, :, :], union[:, 256:, :, :]


        #===================RGB==================
        rgb_kernel_score = self.rgb_conv_cls1(rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4,
                                                                         4)  # [bs,2*5,256,4,4]
        rgb_kernel_regression = self.rgb_conv_r1(rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                            4)  # [bs,4*5,256,4,4]
        rgb_conv_score = self.rgb_conv_cls2(rgb_detection_feature)  # [bs,256,22,22]
        rgb_conv_regression = self.rgb_conv_r2(rgb_detection_feature)  # [bs,256,22,22]
        rgb_conv_scores = rgb_conv_score.reshape(1, -1, self.score_displacement + 4,
                                                 self.score_displacement + 4)  # [1,bsx256,22,22]

        rgb_score_filters = rgb_kernel_score.reshape(-1, 256, 4, 4)  # [bsx10,256,4,4]
        rgb_pred_score = F.conv2d(rgb_conv_scores, rgb_score_filters, groups=N).reshape(N, 10,
                                                                                        self.score_displacement + 1,
                                                                                        self.score_displacement + 1)
        # bs,10,19,19
        rgb_conv_reg = rgb_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        # bs,256,22,22
        rgb_reg_filters = rgb_kernel_regression.reshape(-1, 256, 4, 4)

        rgb_pred_regression = self.rgb_regress_adjust(
            F.conv2d(rgb_conv_reg, rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                      self.score_displacement + 1))

        # ===================ATTN-RGB==================
        attn_rgb_kernel_score = self.attn_rgb_conv_cls1(attn_rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_rgb_kernel_regression = self.attn_rgb_conv_r1(attn_rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        attn_rgb_conv_score = self.attn_rgb_conv_cls2(attn_rgb_detection_feature)
        attn_rgb_conv_regression = self.attn_rgb_conv_r2(attn_rgb_detection_feature)
        attn_rgb_conv_scores = attn_rgb_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)

        attn_rgb_score_filters = attn_rgb_kernel_score.reshape(-1, 256, 4, 4)
        attn_rgb_pred_score = F.conv2d(attn_rgb_conv_scores, attn_rgb_score_filters, groups=N).reshape(N, 10,
                                                                                        self.score_displacement + 1,
                                                                                        self.score_displacement + 1)

        attn_rgb_conv_reg = attn_rgb_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)

        attn_rgb_reg_filters = attn_rgb_kernel_regression.reshape(-1, 256, 4, 4)

        attn_rgb_pred_regression = self.attn_rgb_regress_adjust(
            F.conv2d(attn_rgb_conv_reg, attn_rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                      self.score_displacement + 1))

        # ===================T==================
        t_kernel_score = self.t_conv_cls1(t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        t_conv_score = self.t_conv_cls2(t_detection_feature)
        t_conv_scores = t_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        t_score_filters = t_kernel_score.reshape(-1, 256, 4, 4)  # bsx10,256,4,4
        t_pred_score = F.conv2d(t_conv_scores, t_score_filters, groups=N).reshape(N, 10,
                                                                                  self.score_displacement + 1,
                                                                                  self.score_displacement + 1)

        t_kernel_regression = self.t_conv_r1(t_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                            4)  # bs,4*5,256,4,4
        t_reg_filters = t_kernel_regression.reshape(-1, 256, 4, 4)
        t_conv_regression = self.t_conv_r2(t_detection_feature)  # bs,256,22,22
        t_conv_reg = t_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)

        t_pred_regression = self.t_regress_adjust(
            F.conv2d(t_conv_reg, t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                      self.score_displacement + 1))
        # ===================ATTN-T==================
        attn_t_kernel_score = self.attn_t_conv_cls1(attn_t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_t_conv_score = self.attn_t_conv_cls2(attn_t_detection_feature)
        attn_t_conv_scores = attn_t_conv_score.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)
        attn_t_score_filters = attn_t_kernel_score.reshape(-1, 256, 4, 4)
        attn_t_pred_score = F.conv2d(attn_t_conv_scores, attn_t_score_filters, groups=N).reshape(N, 10,
                                                                                  self.score_displacement + 1,
                                                                                  self.score_displacement + 1)

        attn_t_kernel_regression = self.attn_t_conv_r1(attn_t_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                      4)
        attn_t_reg_filters = attn_t_kernel_regression.reshape(-1, 256, 4, 4)
        attn_t_conv_regression = self.attn_t_conv_r2(attn_t_detection_feature)
        attn_t_conv_reg = attn_t_conv_regression.reshape(1, -1, self.score_displacement + 4, self.score_displacement + 4)

        attn_t_pred_regression = self.attn_t_regress_adjust(
            F.conv2d(attn_t_conv_reg, attn_t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                  self.score_displacement + 1))

        return rgb_pred_score + attn_t_pred_score, attn_rgb_pred_score + t_pred_score, \
               rgb_pred_regression + attn_t_pred_regression, attn_rgb_pred_regression + t_pred_regression

        # return rgb_pred_score ,  t_pred_score, \
        #         rgb_pred_regression ,  t_pred_regression

    def track_init(self, rgb_template, t_template):
        N = rgb_template.size(0)

        rgb_template = self.former_3_layers_featureExtract(rgb_template)
        t_template = self.former_3_layers_featureExtract(t_template)
        rgb_template_feature = self.rgb_featureExtract(rgb_template)  # 输出 [1, 256, 6, 6]
        t_template_feature = self.t_featureExtract(t_template)  # 输出 [1, 256, 6, 6]

        attn_rgb_template_feature = self.attn_rgb_featureExtract(rgb_template)
        attn_rgb_template_feature = self.template_attention_block(attn_rgb_template_feature)
        attn_t_template_feature = self.attn_t_featureExtract(t_template)
        attn_t_template_feature = self.template_attention_block(attn_t_template_feature)


        #attn_rgb_template_feature, attn_t_template_feature = self.template_attention_block(attn_rgb_template_feature,attn_t_template_feature)

        attn_rgb_template_feature, attn_t_template_feature = self.template_attention_block1(attn_rgb_template_feature,attn_t_template_feature)


        # kernel_score=1,2x5,256,4,4   kernel_regression=1,4x5, 256,4,4
        rgb_kernel_score = self.rgb_conv_cls1(rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4,
                                                                         4)  # [1, 10, 256, 4, 4]
        t_kernel_score = self.t_conv_cls1(t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        self.rgb_score_filters = rgb_kernel_score.reshape(-1, 256, 4, 4)  # 2x5, 256, 4, 4
        self.t_score_filters = t_kernel_score.reshape(-1, 256, 4, 4)
        rgb_kernel_regression = self.rgb_conv_r1(rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                            4)  # [1, 20, 256, 4, 4]
        t_kernel_regression = self.t_conv_r1(t_template_feature).view(N, 4 * self.anchor_num, 256, 4,
                                                                            4)  # [1, 20, 256, 4, 4]
        self.rgb_reg_filters = rgb_kernel_regression.reshape(-1, 256, 4, 4)  # 4x5, 256, 4, 4
        self.t_reg_filters = t_kernel_regression.reshape(-1, 256, 4, 4)  # 4x5, 256, 4, 4

        attn_rgb_kernel_score = self.attn_rgb_conv_cls1(attn_rgb_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        attn_t_kernel_score = self.attn_t_conv_cls1(attn_t_template_feature).view(N, 2 * self.anchor_num, 256, 4, 4)
        self.attn_rgb_score_filters = attn_rgb_kernel_score.reshape(-1, 256, 4, 4)
        self.attn_t_score_filters = attn_t_kernel_score.reshape(-1, 256, 4, 4)
        attn_rgb_kernel_regression = self.attn_rgb_conv_r1(attn_rgb_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        attn_t_kernel_regression = self.attn_t_conv_r1(attn_t_template_feature).view(N, 4 * self.anchor_num, 256, 4, 4)
        self.attn_rgb_reg_filters = attn_rgb_kernel_regression.reshape(-1, 256, 4, 4)
        self.attn_t_reg_filters = attn_t_kernel_regression.reshape(-1, 256, 4, 4)


    def track(self, rgb_detection, t_detection):
        N = rgb_detection.size(0)
        rgb_detection = self.former_3_layers_featureExtract(rgb_detection)
        t_detection = self.former_3_layers_featureExtract(t_detection)

        rgb_detection_feature = self.rgb_featureExtract(rgb_detection)  # 1,256,24,24
        t_detection_feature = self.t_featureExtract(t_detection)

        attn_rgb_detection_feature = self.attn_rgb_featureExtract(rgb_detection)  # 1,256,24,24
        attn_rgb_detection_feature = self.detection_attention_block(attn_rgb_detection_feature)
        attn_t_detection_feature = self.attn_t_featureExtract(t_detection)
        attn_t_detection_feature = self.detection_attention_block(attn_t_detection_feature)


        #union = torch.cat((attn_rgb_detection_feature, attn_t_detection_feature), 1)
        #attn_rgb_detection_feature, attn_t_detection_feature = self.detection_attention_block(attn_rgb_detection_feature, attn_t_detection_feature)
        attn_rgb_detection_feature, attn_t_detection_feature = self.detection_attention_block1(attn_rgb_detection_feature, attn_t_detection_feature)

        rgb_conv_score = self.rgb_conv_cls2(rgb_detection_feature)
        t_conv_score = self.t_conv_cls2(t_detection_feature)
        rgb_conv_regression = self.rgb_conv_r2(rgb_detection_feature)
        t_conv_regression = self.t_conv_r2(t_detection_feature)

        rgb_conv_scores = rgb_conv_score.reshape(1, -1, self.score_displacement + 4,
                                                 self.score_displacement + 4)  # [1, 256, 22, 22]
        rgb_pred_score = F.conv2d(rgb_conv_scores, self.rgb_score_filters, groups=N).reshape(N, 10,
                                                                                             self.score_displacement + 1,
                                                                                             self.score_displacement + 1)  # [1, 10, 19, 19], self.score_filters.shape ==[10,256, 4, 4]
        t_conv_scores = t_conv_score.reshape(1, -1, self.score_displacement + 4,
                                             self.score_displacement + 4) # [1, 256, 22, 22]
        t_pred_score = F.conv2d(t_conv_scores, self.t_score_filters, groups=N).reshape(N, 10,
                                                                                       self.score_displacement + 1,
                                                                                       self.score_displacement + 1)
        rgb_conv_reg = rgb_conv_regression.reshape(1, -1, self.score_displacement + 4,
                                                   self.score_displacement + 4)  # [1, 256, 22, 22]
        t_conv_reg = t_conv_regression.reshape(1, -1, self.score_displacement + 4,
                                                   self.score_displacement + 4)  # [1, 256, 22, 22]
        rgb_pred_regression = self.rgb_regress_adjust(
            F.conv2d(rgb_conv_reg, self.rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                           self.score_displacement + 1))  # [1, 20, 19, 19]
        t_pred_regression = self.t_regress_adjust(
            F.conv2d(t_conv_reg, self.t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                           self.score_displacement + 1))  # [1, 20, 19, 19]

        attn_rgb_conv_score = self.attn_rgb_conv_cls2(
            attn_rgb_detection_feature)
        attn_t_conv_score = self.attn_t_conv_cls2(attn_t_detection_feature)
        attn_rgb_conv_regression = self.attn_rgb_conv_r2(
            attn_rgb_detection_feature)
        attn_t_conv_regression = self.attn_t_conv_r2(attn_t_detection_feature)

        attn_rgb_conv_scores = attn_rgb_conv_score.reshape(1, -1, self.score_displacement + 4,
                                                 self.score_displacement + 4)
        attn_rgb_pred_score = F.conv2d(attn_rgb_conv_scores, self.attn_rgb_score_filters, groups=N).reshape(N, 10,
                                                                                             self.score_displacement + 1,
                                                                                             self.score_displacement + 1)
        attn_t_conv_scores = attn_t_conv_score.reshape(1, -1, self.score_displacement + 4,
                                             self.score_displacement + 4)
        attn_t_pred_score = F.conv2d(attn_t_conv_scores, self.attn_t_score_filters, groups=N).reshape(N, 10,
                                                                                       self.score_displacement + 1,
                                                                                       self.score_displacement + 1)
        attn_rgb_conv_reg = attn_rgb_conv_regression.reshape(1, -1, self.score_displacement + 4,
                                                   self.score_displacement + 4)
        attn_t_conv_reg = attn_t_conv_regression.reshape(1, -1, self.score_displacement + 4,
                                               self.score_displacement + 4)
        attn_rgb_pred_regression = self.attn_rgb_regress_adjust(
            F.conv2d(attn_rgb_conv_reg, self.attn_rgb_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                           self.score_displacement + 1))
        attn_t_pred_regression = self.attn_t_regress_adjust(
            F.conv2d(attn_t_conv_reg, self.attn_t_reg_filters, groups=N).reshape(N, 20, self.score_displacement + 1,
                                                                       self.score_displacement + 1))


        # return rgb_pred_score + attn_t_pred_score, attn_rgb_pred_score + t_pred_score, \
        #        rgb_pred_regression + t_pred_regression

        #attn_rgb_pred_score_np = torch.mean(attn_rgb_conv_scores, dim=1).squeeze().detach().cpu().numpy()
        # rgb_pred_score_np = torch.mean(rgb_conv_reg, dim=1).squeeze().detach().cpu().numpy()
        # attn_t_pred_score_np = torch.mean(t_pred_regression, dim=1).squeeze().detach().cpu().numpy()
        # t_pred_score_np = torch.mean(t_conv_score, dim=1).squeeze().detach().cpu().numpy()
        from scipy.ndimage import zoom
        #放大5倍，避免网格感
        # attn_rgb_pred_score_np = zoom(attn_rgb_pred_score_np, 5, order=3)
        # cm1 = plt.cm.get_cmap('jet')
        # plt.imshow(attn_rgb_pred_score_np, cmap=cm1, interpolation='bicubic')  # Reverse the colormap
        # plt.title('', fontsize=10)
        # plt.colorbar()

        # attn_rgb_pred_score_np = attn_rgb_conv_scores[0, 0].detach().cpu().numpy()  # 取第一个 batch，第一通道
        # attn_rgb_pred_score_np = attn_rgb_conv_scores[0].mean(axis=0).detach().cpu().numpy()  # 对通道求平均
        # attn_rgb_pred_score_np = attn_rgb_conv_scores[0].mean(dim=0).detach().cpu().numpy()  # 取最大响应
        # rgb_pred_score_np = rgb_conv_reg[0].mean(dim=0).detach().cpu().numpy()  # 取最大响应
        # attn_t_pred_score_np = t_pred_regression[0].mean(dim=0).detach().cpu().numpy()  # 取最大响应
        # t_pred_score_np = t_conv_score[0].mean(dim=0).detach().cpu().numpy()  # 取最大响应
        # # 进行插值放大，避免马赛克
        # attn_rgb_pred_score_np = zoom(attn_rgb_pred_score_np, 4, order=3)
        # rgb_pred_score_np = zoom(rgb_pred_score_np, 4, order=3)
        # attn_t_pred_score_np = zoom(attn_t_pred_score_np, 4, order=3)
        # t_pred_score_np = zoom(t_pred_score_np, 4, order=3)
        #
        # #plt.figure(figsize=(8, 6), dpi=300)  # 提高 dpi 以提升清晰度
        # plt.imshow(attn_rgb_pred_score_np, cmap='jet', interpolation='bicubic', extent=[0, 0.001, 0, 0.001])
        # plt.title('', fontsize=10)
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(rgb_pred_score_np, cmap='jet', interpolation='bicubic', extent=[0, 0.001, 0, 0.001])
        # plt.title('', fontsize=10)
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(attn_t_pred_score_np, cmap='jet', interpolation='bicubic', extent=[0, 0.001, 0, 0.001])
        # plt.title('', fontsize=10)
        # plt.colorbar()
        # plt.show()
        #
        # plt.imshow(t_pred_score_np, cmap='jet', interpolation='bicubic', extent=[0, 0.001, 0, 0.001])
        # plt.title('', fontsize=10)
        # plt.colorbar()
        # plt.show()

        # attn_rgb_pred_score_np = torch.mean(attn_rgb_conv_reg, dim=1).squeeze().detach().cpu().numpy()
        # rgb_pred_score_np = torch.mean(rgb_conv_score, dim=1).squeeze().detach().cpu().numpy()
        # attn_t_pred_score_np = torch.mean(attn_t_conv_reg, dim=1).squeeze().detach().cpu().numpy()
        # t_pred_score_np = torch.mean(t_conv_score, dim=1).squeeze().detach().cpu().numpy()

        # Plot the attention score# Replace with the actual size of your input image
        input_height, input_width = 256, 256

        # Assuming attn_t_pred_score_np and rgb_pred_score_np are the same size as the input image

        # # Plot the attention score
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 4, 1)
        # cm1 = plt.cm.get_cmap('jet')
        # plt.imshow(attn_rgb_pred_score_np, cmap=cm1)  # Reverse the colormap
        # plt.title('', fontsize=10)
        # plt.colorbar()
        # #
        # # # Plot the RGB score
        # plt.subplot(1, 4, 2)
        # cm2 = plt.cm.get_cmap('jet')
        # plt.imshow(attn_t_pred_score_np, cmap=cm2)  # Reverse the colormap
        # plt.title('', fontsize=10)
        # plt.colorbar()  # Set the orientation to 'vertical'
        #
        # # Plot the attention score
        # plt.subplot(1, 4, 3)
        # cm3 = plt.cm.get_cmap('jet')
        # plt.imshow(rgb_pred_score_np, cmap=cm3)  # Reverse the colormap
        # plt.title('', fontsize=10)
        # plt.colorbar()
        # #
        # # # Plot the RGB score
        # plt.subplot(1, 4, 4)
        # cm4 = plt.cm.get_cmap('jet')
        # plt.imshow(t_pred_score_np, cmap=cm4)  # Reverse the colormap
        # plt.title('', fontsize=10)
        # plt.colorbar()  # Set the orientation to 'vertical'
        # #
        # # # Show the plots
        # plt.show()

        # attn_rgb_conv_reg_np = torch.mean(attn_rgb_conv_reg, dim=1).squeeze().detach().cpu().numpy()
        # attn_rgb_detection_feature_np = torch.mean(attn_rgb_detection_feature, dim=1).squeeze().detach().cpu().numpy()
        # attn_t_conv_reg_np = torch.mean(attn_t_conv_reg, dim=1).squeeze().detach().cpu().numpy()
        # attn_t_detection_feature_np = torch.mean(attn_t_detection_feature, dim=1).squeeze().detach().cpu().numpy()


        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, 4, 1)
        # #ax=sns.heatmap(attn_rgb_pred_score_np, cmap="viridis")
        # ax = sns.heatmap(attn_rgb_pred_score_np, cmap='viridis', cbar=False)
        # ax.set_title('', fontsize=10)
        # plt.colorbar(ax.collections[0])
        # ax.set_aspect('equal')  # 设置为正方形
        # ax.set_xticks([])  # 去掉x轴刻度
        # ax.set_yticks([])  # 去掉y轴刻度
        # ax.set_frame_on(False)  # 去掉轴框
        #
        # plt.subplot(1, 4, 2)
        # #ax=sns.heatmap(attn_t_pred_score_np, cmap="viridis")
        # ax = sns.heatmap(attn_t_pred_score_np, cmap='viridis', cbar=False)
        # ax.set_title('', fontsize=10)
        # plt.colorbar(ax.collections[0])
        # ax.set_aspect('equal')  # 设置为正方形
        # ax.set_xticks([])  # 去掉x轴刻度
        # ax.set_yticks([])  # 去掉y轴刻度
        # ax.set_frame_on(False)  # 去掉轴框
        #
        # plt.subplot(1, 4, 3)
        # #ax=sns.heatmap(rgb_pred_score_np, cmap="viridis")
        # ax = sns.heatmap(rgb_pred_score_np, cmap='viridis', cbar=False)
        # ax.set_title('', fontsize=10)
        # plt.colorbar(ax.collections[0])
        # ax.set_aspect('equal')  # 设置为正方形
        # ax.set_xticks([])  # 去掉x轴刻度
        # ax.set_yticks([])  # 去掉y轴刻度
        # ax.set_frame_on(False)  # 去掉轴框
        #
        # plt.subplot(1, 4, 4)
        # #ax=sns.heatmap(t_pred_score_np, cmap="viridis")
        # ax = sns.heatmap(t_pred_score_np,  cmap='viridis', cbar=False)
        # ax.set_title('', fontsize=10)
        # plt.colorbar(ax.collections[0])
        # ax.set_aspect('equal')  # 设置为正方形
        # ax.set_xticks([])  # 去掉x轴刻度
        # ax.set_yticks([])  # 去掉y轴刻度
        # ax.set_frame_on(False)  # 去掉轴框
        #
        # plt.show()



        return rgb_pred_score + attn_t_pred_score, attn_rgb_pred_score + t_pred_score, \
               rgb_pred_regression + t_pred_regression

        # return rgb_pred_score ,  t_pred_score, \
        #        rgb_pred_regression + t_pred_regression



class DepthwiseSeparableConv(nn.Module):
    # 类的初始化方法
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        # 调用父类的初始化方法
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层，使用与输入通道数相同的组数，使每个输入通道独立卷积
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels),
                                       nn.BatchNorm2d(in_channels),
                                       # 激活函数层，使用LeakyReLU
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )
        # 逐点卷积层，使用1x1卷积核进行卷积，以改变通道数
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                       nn.BatchNorm2d(out_channels),
                                       # 激活函数层，使用LeakyReLU
                                       nn.LeakyReLU(0.1, inplace=True)
                                       )

    # 定义前向传播方法
    def forward(self, x):
        # 输入x通过深度卷积层
        x = self.depthwise(x)
        # 经过深度卷积层处理后的x通过逐点卷积层
        x = self.pointwise(x)
        # 返回最终的输出
        return x

def count_param(model):
    param_count=0
    for param in model.parameters():
        param_count+=param.view(-1).size()[0]
    return param_count



# if __name__=='__main__':
#     model = SiamCCANet()
#     param=count_param(model)
#     print('SiamCCANet total parameters: %.2fM (%d)' % (param/1e6,param))