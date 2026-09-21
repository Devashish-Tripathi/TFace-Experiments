# Modifications Copyright (C) 2026 Devashish Tripathi
# Originally licensed under Apache 2.0 by Tencent.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchjpeg import dct

def images_to_batch(x):
    x = (x + 1) / 2 * 255
    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    if x.shape[1] != 3:
        print("Wrong input, Channel should equals to 3")
        return
    x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
    x -= 128
    bs, ch, h, w = x.shape
    block_num = h // 8
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
                 stride=(8, 8))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, 8, 8)
    dct_block = dct.block_dct(x)
    dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
    dct_block = dct_block[:, :, 1:, :, :]  # remove DC
    dct_block = dct_block.reshape(bs, -1, block_num, block_num)
    return dct_block


class NoisyActivation(nn.Module):
    def __init__(self, input_shape=112, 
                 budget_mean=4, 
                 sensitivity=False, 
                 given_locs= None, 
                 donot_use_loc= True,
                 sens_pth = None):
        
        super(NoisyActivation, self).__init__()
        self.h, self.w = input_shape, input_shape
        self.fix_shape = (189, self.h, self.w)
        self.size = 189 * self.h * self.w
        self.donot_use_loc = donot_use_loc

        # sensitivity
        if sensitivity is False:
            print("defaulting to ones")
            sensitivity = torch.ones(self.fix_shape).cuda()
        elif sens_pth is None or not os.path.exists(sens_pth):
            print("No path found, defaulting to ones")
            sensitivity = torch.ones(self.fix_shape).cuda()
        else:
            sensitivity = torch.load(sens_pth).cuda()
        self.sensitivity = sensitivity.reshape(self.size)

        # learnable budget params
        self.rhos = nn.Parameter(torch.zeros(self.fix_shape))
        self.budget = budget_mean * self.size
        self.laplace = torch.distributions.laplace.Laplace(0, 1)
        self.rhos.requires_grad = True

        # locations. Conflicts with what is established in paper, as it is zero-centered there. Made optional.
        if self.donot_use_loc:
            print("not using locs")
        if not self.donot_use_loc:
            if given_locs is None:
                self.given_locs = torch.zeros(self.fix_shape)
            else:
                self.given_locs = given_locs
            # print(self.fix_shape)
            # print(self.given_locs.shape)
            self.locs = nn.Parameter(self.given_locs.clone().detach())
            self.locs.requires_grad = True

    def scales(self):
        # changed
        softmax = nn.Softmax(dim=-1)
        return (self.sensitivity / (softmax(self.rhos.reshape(189 * self.h * self.w)) * self.budget)).reshape(189, self.h, self.w)

    def sample_noise(self):
        epsilon = self.laplace.sample(self.fix_shape).cuda()
        if not self.donot_use_loc:
            return self.locs + self.scales() * epsilon
        return self.scales() * epsilon

    def forward(self, input):
        noise = self.sample_noise()
        output = input + noise
        return output

# from TFace/recognition/tasks/PartialFace
def idct_transform(x, size=8, stride=8, pad=0, dilation=1, ratio=8):
    """
        The inverse of DCT transform.
        Transform frequency channels (must be 192 channels, can be padded with 0) back to the spatial image.
    """

    b, _, h, w = x.shape

    x = x.view(b, 3, 64, h, w)
    x = x.permute(0, 1, 3, 4, 2)
    x = x.view(b, 3, h * w, 8, 8)
    x = dct.block_idct(x)
    x = x.view(b * 3, h * w, 64)
    x = x.transpose(1, 2)
    x = F.fold(x, output_size=(112 * ratio, 112 * ratio),
               kernel_size=(size, size), dilation=dilation, padding=pad, stride=(stride, stride))
    x = x.view(b, 3, 112 * ratio, 112 * ratio)
    x = x + 128
    x = dct.to_rgb(x)
    x = x / 255
    x = F.interpolate(x, scale_factor=1 / ratio, mode='bilinear', align_corners=True)
    x = x.clamp(min=0.0, max=1.0)
    return x
