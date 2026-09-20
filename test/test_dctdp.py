# Modifications Copyright (C) 2026 Devashish Tripathi
# Originally licensed under Apache 2.0 by Tencent.


import os
import sys
import argparse
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torchkit.backbone import get_model
from torchjpeg import dct

class NoisyActivation(nn.Module):
    def __init__(self, input_shape=112, budget_mean=4, sensitivity=None):
        super(NoisyActivation, self).__init__()
        self.h, self.w = input_shape, input_shape
        if sensitivity is None:
            sensitivity = torch.ones([189, self.h, self.w]).cuda()
        self.sensitivity = sensitivity.reshape(189 * self.h * self.w)
        self.given_locs = torch.zeros((189, self.h, self.w))
        size = self.given_locs.shape
        self.budget = budget_mean * 189 * self.h * self.w
        self.locs = nn.Parameter(torch.Tensor(size).copy_(self.given_locs))
        self.rhos = nn.Parameter(torch.zeros(size))
        self.laplace = torch.distributions.laplace.Laplace(0, 1)
        # self.rhos.requires_grad = True
        # self.locs.requires_grad = True

    def scales(self):
        # changed
        softmax = nn.Softmax(dim=-1)
        return (self.sensitivity / (softmax(self.rhos.reshape(189 * self.h * self.w))
                * self.budget)).reshape(189, self.h, self.w)

    def sample_noise(self):
        epsilon = self.laplace.sample(self.rhos.shape).cuda()
        return self.locs + self.scales() * epsilon

    def forward(self, input):
        noise = self.sample_noise()
        output = input + noise
        return output

    # def aux_loss(self):
    #     scale = self.scales()
    #     loss = -1.0 * torch.log(scale.mean())
    #     return loss


def images_to_batch(x):
    x = (x + 1) / 2 * 255
    x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
    if x.shape[1] != 3:
        raise ValueError("Wrong input, Channel should equals to 3")

    x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
    x -= 128
    bs, ch, h, w = x.shape
    block_num = h // 8
    x = x.view(bs * ch, 1, h, w)
    x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0, stride=(8, 8))
    x = x.transpose(1, 2)
    x = x.view(bs, ch, -1, 8, 8)
    dct_block = dct.block_dct(x)
    dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
    dct_block = dct_block[:, :, 1:, :, :]  # remove DC
    dct_block = dct_block.reshape(bs, -1, block_num, block_num)
    return dct_block


class DCTDPModel(nn.Module):
    def __init__(self, backbone, noise_model=None):
        super(DCTDPModel, self).__init__()
        self.backbone = backbone
        self.noise_model = noise_model
    
    def forward(self, x):
        # x is [B, 3, 112, 112]
        x_dct = images_to_batch(x)
        if self.noise_model:
            x_dct = self.noise_model(x_dct)
        features = self.backbone(x_dct)
        return features



def parse_args():
    parser = argparse.ArgumentParser(description= 'DCTDP eval code')
    # there are three checkpoints, this should ideally be enough to pick all three
    # {Backbone, META}_Epoch_X_checkpoint.pth AND HEAD_Epoch_X_Split_Y_checkpoint.pth
    # TODO: for HEAD, need to verify what split means
    parser.add_argument('--ckpt_path', required= True, default= None, help= 'Path to model checkpoints')
    parser.add_argument('--epoch', required= True, default= -1, help= 'Weights of which epoch. Select -1 for latest')
    parser.add_argument('--gpu_ids', default = '0', help= 'GPU IDs; comma separated') # how will it take multiple, assuming you implement it?
    parser.add_argument('--data_root', default='', required=True, help='eval data root') # what kind of path. correlate with train
    parser.add_argument('--out_path', default='./output', help='output path') # what kind of output? how stored? make sure to implement folder
    parser.add_argument('--batch_size', default=64, help='batch size')
    parser.add_argument('--random_seed', default=1337, help='random seed')
    parser.add_argument('--epsilon', default=0.5, help='privacy budget')
    parser.add_argument('--use_noise', , help='privacy budget')
    return parser.parse_args()


def load_checkpoints(ckpt_dir, prefix, epoch):
    # {Backbone, META}_Epoch_X_checkpoint.pth
    path = os.path.join(ckpt_dir, f"{prefix}_Epoch_*_checkpoint.pth")
    files = glob.glob(path)
    if not files:
        return None
    epochs = [int(f.split('_Epoch_')[-1].split('.')[0].split('_')[0]) for f in files]
    if epoch == -1:
        epoch = max(epochs)
    path = os.path.join(ckpt_dir, f"{prefix}_Epoch_{epoch}_checkpoint.pth")
    if os.path.exists(path):
        return path
    return None

def main():
    """
    Perform evaluation on the provided dataset. Any specific thing to follow?

    Take images, transform them, do BDCT, remove DC, add noise, pass through model, get result, compare
    """
    
    # defaults
    args = parse_args()
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    input_size = [112, 112]
    embedding_size = 512
    device = torch.device(f"cuda:{args.gpu_ids.split(',')[0]}" if torch.cuda.is_available() else "cpu")
    
    # load model
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError("Checkpoint Path does NOT exist!")
    bbonpth = load_checkpoints(args.ckpt_path, 'Backbone', args.epoch)
    metapth = load_checkpoints(args.ckpt_path, 'META', args.epoch)
    if not bbonpath or metapth:
        raise RuntimeError("Weights Not Found!")

    backbone = get_model('IR_50')(input_size, input_channel= 189)
    # the DDP module thing? What is it, what code do you need to know if that is there or not and what made you put it here?
    backbone.load_state_dict(torch.load(bbonpath, weights_only= True, map_location="cpu"))
    backbone.eval()

    noise_model = None
    if args.use_noise:
        noise_model = NoisyActivation(input_shape= 112, budget_mean= args.epsilon)
        # loading its weights

        noise_model.eval()
    
    model = DCTDPModel(backbone, noise_model)
    model = model.to(device)
    model.eval()


    # load val data
    # transform data
    # run the validation
    # output/save the results    