import global_variable
base_dir = global_variable.base_dir

import sys
sys.path.append(base_dir)

import torch
import torch.nn as nn
from models.resnet_CBAM import resnet18_cbam
from models import vgg19_bn, vgg19, vgg16
from models import cifar_vgg19_bn,cifar_vgg16, cifar_vgg19

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return out.unsqueeze(2).unsqueeze(3)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(combined))
        return out

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        channel_out = self.channel_attention(x) * x
        spatial_out = self.spatial_attention(channel_out) * channel_out
        return spatial_out

# 创建一个 CBAM 模块并应用于输入图像


class his_equ_net(nn.Module):
    def __init__(self):
        super(his_equ_net, self).__init__()
        self.cifar_spatial = cifar_vgg19_bn(num_classes=2)
        self.cifar_gradient = cifar_vgg19_bn(num_classes=2)
        self.imagenet_spatial = vgg19_bn(num_classes=2)
        self.imagenet_gradient = vgg19_bn(num_classes=2)

    def forward(self, x, grad):
        if x.size(3) == 224:
            logit1 = self.imagenet_spatial(x)
            logit2 = self.imagenet_gradient(grad)
            final_logit = logit1 + logit2
        else:
            logit1 = self.cifar_spatial(x)
            logit2 = self.cifar_gradient(grad)
            final_logit = logit1 + logit2

        return final_logit
    # def forward(self,  x):
    #     if x.size(3) == 224:
    #         logit2 = self.imagenet_spatial(x)
    #     else:
    #         logit2 = self.cifar_spatial(x)
    #
    #     return logit2