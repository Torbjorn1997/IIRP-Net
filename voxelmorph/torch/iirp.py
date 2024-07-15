import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal
from .modelio import LoadableModel, store_config_args
import nibabel as nib
import math

# from . import layers


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        # print('size', size)

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):

        # new locations
        # print('flow', flow.shape)
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvResBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = x + out
        out = self.activation(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_channel=1, first_channel=8):
        super(Encoder, self).__init__()
        c = first_channel
        self.block1 = ConvBlock(in_channel, c)
        self.block2 = ConvBlock(c, c * 2)
        self.block3 = ConvBlock(c *2, c * 4)
        self.block4 = ConvBlock(c *4, c * 4)

    def forward(self, x):
        out1 = self.block1(x)
        x = nn.AvgPool3d(2)(out1)
        out2 = self.block2(x)
        x = nn.AvgPool3d(2)(out2)
        out3 = self.block3(x)
        x = nn.AvgPool3d(2)(out3)
        out4 = self.block4(x)
        return out1, out2, out3, out4

class DecoderBlock(nn.Module):
    def __init__(self, x_channel, y_channel, out_channel):
        super(DecoderBlock, self).__init__()
        self.Conv1 = ConvBlock(x_channel+y_channel, out_channel)
        self.Conv2 = ConvResBlock(out_channel, out_channel)
        self.Conv3 = ConvResBlock(out_channel, out_channel)
        # self.Conv3_2 = ConvResBlock(out_channel, out_channel)
        # self.Conv3_3 = ConvResBlock(out_channel, out_channel)
        self.Conv4 = nn.Conv3d(out_channel, out_channel//2, 3, padding=1)
        self.Conv5 = nn.Conv3d(out_channel//2, 3, 3, padding=1)

    def forward(self, x, y):
        concat = torch.cat([x, y], dim=1)
        cost_vol = self.Conv1(concat)
        cost_vol = self.Conv2(cost_vol)
        cost_vol = self.Conv3(cost_vol)
        # cost_vol = self.Conv3_2(cost_vol)
        # cost_vol = self.Conv3_3(cost_vol)
        cost_vol = self.Conv4(cost_vol)
        flow = self.Conv5(cost_vol)

        return flow

class RPNet(LoadableModel):
    @store_config_args
    def __init__(self, size=(80, 96, 80), in_channel=1, first_channel=8):
        super(RPNet, self).__init__()
        c = first_channel
        self.encoder = Encoder(in_channel, c)
        self.decoder4 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder3 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder2 = DecoderBlock(x_channel = 16, y_channel = 16, out_channel = 32)
        self.decoder1 = DecoderBlock(x_channel = 8, y_channel = 8, out_channel = 16)
        self.size = size

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, y):
        fx1, fx2, fx3, fx4 = self.encoder(x)
        fy1, fy2, fy3, fy4 = self.encoder(y)

        ar = 1
        br = 1
        cr = 1
        dr = 1

        wx4 = fx4

        flowall =  None
        for aa in range(ar):
            flow = self.decoder4(wx4, fy4)
            if aa == 0:
                flowall = flow
            else:
                flowall = self.transformer[3](flowall, flow)+flow

        flowall = self.up(2*flowall)
        for bb in range(br):
            wx3 = self.transformer[2](fx3, flowall)
            flow = self.decoder3(wx3, fy3)
            flowall = self.transformer[2](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        for cc in range(cr):
            wx2 = self.transformer[1](fx2, flowall)
            flow = self.decoder2(wx2, fy2)
            flowall = self.transformer[1](flowall, flow) + flow

        flowall = self.up(2 * flowall)
        for dd in range(dr):
            wx1 = self.transformer[0](fx1, flowall)
            flow = self.decoder1(wx1, fy1)
            flowall = self.transformer[0](flowall, flow) + flow
        warped_x = self.transformer[0](x, flowall)

        return warped_x, flowall

class IIRPNet(LoadableModel):
    @store_config_args
    def __init__(self, size=(80, 96, 80), in_channel=1, first_channel=8):
        super(IIRPNet, self).__init__()
        c = first_channel
        self.encoder = Encoder(in_channel, c)
        self.decoder4 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder3 = DecoderBlock(x_channel = 32, y_channel = 32, out_channel = 32)
        self.decoder2 = DecoderBlock(x_channel = 16, y_channel = 16, out_channel = 32)
        self.decoder1 = DecoderBlock(x_channel = 8, y_channel = 8, out_channel = 16)
        self.size = size

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in size]))
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def normalized_cross_correlation(self, img1, img2):
        # 计算均值
        mean_img1 = torch.mean(img1)
        mean_img2 = torch.mean(img2)

        # 计算标准差
        std_img1 = torch.std(img1)
        std_img2 = torch.std(img2)

        # 计算NCC
        ncc = torch.mean((img1 - mean_img1) * (img2 - mean_img2) / (std_img1 * std_img2))

        return ncc

    def pnsr(self, original_image, processed_image):
        # 确保输入张量的数据类型为浮点数
        original_image = original_image.float()
        processed_image = processed_image.float()

        # 计算均方误差（MSE）
        mse = torch.mean((original_image - processed_image) ** 2)

        # 如果MSE为0，PSNR无穷大，这里我们使用一个很小的非零值替代0以避免错误
        if mse == 0:
            return float('inf')

        # 计算PSNR
        max_intensity = 1.0  # 假设图像像素值的范围在0到255之间
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)

        return psnr

    def forward(self, x, y):
        fx1, fx2, fx3, fx4 = self.encoder(x)
        fy1, fy2, fy3, fy4 = self.encoder(y)

        ar = 10
        br = 10
        cr = 10
        dr = 10

        pa = pb = pc = pd = 0

        current_iter = []
        wx4 = fx4
        mse_a = 100 #mse mae
        ncc_a = 0 #ncc,pnsr
        delta1 =0.005
        delta2 =0.005
        delta3 =0.005
        delta4 =0.005

        flowall =  None
        for aa in range(ar):
            flow = self.decoder4(wx4, fy4)
            previous_flow = flowall
            if aa == 0:
                flowall = flow
            else:
                flowall = self.transformer[3](flowall, flow)+flow
            wx4 = self.transformer[3](fx4, flowall)
            # not use in train
            flowx4 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)(8*flowall)
            mx4 = self.transformer[0](x, flowx4)
            ncc = self.normalized_cross_correlation(mx4, y)
            ig = ncc < (ncc_a + delta1)
            if ig:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pa += 1
        current_iter.append(pa)

        flowall = self.up(2*flowall)
        mse_a = 100
        ncc_a = 0
        previous_flow = flowall
        for bb in range(br):
            previous_flow = flowall
            wx3 = self.transformer[2](fx3, flowall)
            # not use in train
            flowx3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)(4*flowall)
            mx3 = self.transformer[0](x, flowx3)
            ncc = self.normalized_cross_correlation(mx3, y)
            ig = ncc < (ncc_a + delta1)
            if ig:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pb += 1

            flow = self.decoder3(wx3, fy3)
            previous_flow = flowall
            flowall = self.transformer[2](flowall, flow) + flow
        current_iter.append(pb)

        flowall = self.up(2 * flowall)
        previous_flow = flowall
        mse_a = 100
        ncc_a = 0
        for cc in range(cr):
            previous_flow = flowall
            wx2 = self.transformer[1](fx2, flowall)
            # not use in train
            flowx2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)(2*flowall)
            mx2 = self.transformer[0](x, flowx2)
            ncc = self.normalized_cross_correlation(mx2, y)
            ig = ncc < (ncc_a + delta1)
            if ig:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pc += 1

            flow = self.decoder2(wx2, fy2)
            previous_flow = flowall
            flowall = self.transformer[1](flowall, flow) + flow
        current_iter.append(pc)

        flowall = self.up(2 * flowall)
        previous_flow = flowall
        mse_a = 100
        ncc_a = 0
        for dd in range(dr):
            previous_flow = flowall
            wx1 = self.transformer[0](fx1, flowall)
            # not use in train
            mx = self.transformer[0](x, flowall)
            ncc = self.normalized_cross_correlation(mx, y)
            ig = ncc < (ncc_a + delta1)
            if ig:
                flowall = previous_flow
                break
            else:
                ncc_a = ncc
                pd += 1

            flow = self.decoder1(wx1, fy1)
            previous_flow = flowall
            flowall = self.transformer[0](flowall, flow) + flow
        current_iter.append(pd)

        warped_x = self.transformer[0](x, flowall)

        return warped_x, flowall, current_iter
