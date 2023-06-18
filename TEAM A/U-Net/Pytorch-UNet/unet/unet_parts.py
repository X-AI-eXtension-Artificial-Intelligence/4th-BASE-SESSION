""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

# 각각의 컨볼루션 레이어 뒤에 Batch Normalization과 ReLU 활성화 함수를 적용
# U-Net 모델에서 encoder 부분에서 반복적으로 사용
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(mid_channels), # Batch Normalization
            nn.ReLU(inplace=True), # ReLU
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), # Batch Normalization
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) # ReLU
        )

    def forward(self, x): # 입력 텐서를 받고 두 개의 컨볼루션 레이어를 적용한 뒤 출력
        return self.double_conv(x)


class Down(nn.Module): # input imamge를 downsampling하여 size를 줄이는 down block을 구현
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), # input imamge를 2배로 downsampling
            DoubleConv(in_channels, out_channels) # downsampling한 image에 convolution layer와 batch norm, ReLU를 두 번 적용
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module): # input imamge를 upsampling하여 size를 늘리는 up block을 구현
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # input imamge를 2배로 upsampling
            # downsampling한 input image와 같은 크기를 갖도록 padding을 적용한 후 이를 concatenate
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else: # 3차원 이미지를 소화하기 위해서 추가
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2) # 특정 stride와 kernel size로 upsampling
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2): # 각각 up module, down module
        x1 = self.up(x1)
        # input is CHW
        # x2와 x1의 크기 차이를 계산
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # x1과 x2의 크기가 다르면, 두 개의 텐서를 합치기 위해 x1을 padding
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module): # U-Net의 출력 채널을 생성
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) # input image를 output image로 변환

    def forward(self, x):
        return self.conv(x)
