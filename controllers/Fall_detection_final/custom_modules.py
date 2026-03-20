import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

# 注意力机制模块
class CBAM(nn.Module):
    """ Convolutional Block Attention Module """

    def __init__(self, c1, reduction_ratio=16):
        super().__init__()
        c2 = max(c1 // reduction_ratio, 1)

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c2, 1),
            nn.ReLU(),
            nn.Conv2d(c2, c1, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca

        # 空间注意力
        sa_in = torch.cat([torch.mean(x_ca, dim=1, keepdim=True),
                           torch.max(x_ca, dim=1, keepdim=True)[0]], dim=1)
        sa = self.spatial_attention(sa_in)
        return x_ca * sa


# 坐标注意力
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, oup, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, oup, 1, bias=False)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_w * a_h


# 加权特征金字塔网络
class BiFPN(nn.Module):
    """ Bi-directional Feature Pyramid Network """

    def __init__(self, c3, c4, c5, out_channels=256, num_layers=3):
        super().__init__()
        self.out_channels = out_channels
        self.conv6 = Conv(c5, out_channels, 1)
        self.conv5 = Conv(c4, out_channels, 1)
        self.conv4 = Conv(c3, out_channels, 1)

        # 特征融合层
        self.fusions = nn.ModuleList()
        for _ in range(num_layers):
            self.fusions.append(nn.Sequential(
                Conv(out_channels * 3, out_channels, 1),
                nn.ReLU()
            ))

    def forward(self, inputs):
        # inputs: [p3, p4, p5]
        p3, p4, p5 = inputs

        # 下采样路径
        p4_in = self.conv5(p4)
        p5_in = self.conv6(p5)

        # 上采样融合
        p5_up = nn.functional.interpolate(p5_in, scale_factor=2, mode='nearest')
        p4_fused = p4_in + p5_up

        p4_up = nn.functional.interpolate(p4_fused, scale_factor=2, mode='nearest')
        p3_fused = self.conv4(p3) + p4_up

        # 下采样融合
        p3_down = nn.functional.max_pool2d(p3_fused, kernel_size=3, stride=2, padding=1)
        p4_fused = p4_fused + p3_down

        p4_down = nn.functional.max_pool2d(p4_fused, kernel_size=3, stride=2, padding=1)
        p5_fused = p5_in + p4_down

        # 加权融合
        features = torch.cat([
            p3_fused,
            nn.functional.interpolate(p4_fused, scale_factor=2, mode='nearest'),
            nn.functional.interpolate(p5_fused, scale_factor=4, mode='nearest')
        ], dim=1)

        # 多级融合
        for fusion in self.fusions:
            features = fusion(features)

        return features


# 空模块 (用于条件判断)
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x