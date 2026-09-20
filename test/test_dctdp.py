# Author: Devashish Tripathi
# Based on codes originally licensed under Apache 2.0 by Tencent.


import os
import argparse
import glob
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torchkit.backbone import get_model
from torchjpeg import dct
from utils import perform_val_bin, get_val_pair_from_bin

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

    def scales(self):
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
    parser.add_argument('--ckpt_path', required= True, default= None, help= 'Path to folder containing model checkpoints')
    parser.add_argument('--epoch', default= -1, help= 'Weights of which epoch. Select -1 for latest')
    parser.add_argument('--gpu_ids', default = '0', help= 'GPU IDs; comma separated')
    parser.add_argument('--data_root', default='', required=True, help='eval data root') 
    parser.add_argument('--bin_name', default='', required=True, help='name of bin file to eval. use conv script if needed.') 
    # parser.add_argument('--out_path', default='./output', help='output path')
    parser.add_argument('--model_name', default='test', help='name of model')
    parser.add_argument('--data_name', default='sample', help='name of eval dataset')
    parser.add_argument('--batch_size', default=64, help='batch size')
    parser.add_argument('--random_seed', default=1337, help='random seed')
    parser.add_argument('--epsilon', default=0.5, help='privacy budget')
    parser.add_argument('--use_noise', action='store_true' , help='whether to use noise model')
    return parser.parse_args()

def main():
    """
    Perform evaluation on the provided dataset.
    """
    
    # defaults
    args = parse_args()
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    input_size = [112, 112]
    device = torch.device(f"cuda:{args.gpu_ids.split(',')[0]}" if torch.cuda.is_available() else "cpu")
    
    # load model
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError("Checkpoint Path does NOT exist!")

    # {Backbone, Noise}_Epoch_X_checkpoint.pth
    path = os.path.join(args.ckpt_path, f"Backbone_Epoch_*_checkpoint.pth")
    files = glob.glob(path)
    if not files:
        raise RuntimeError("Checkpoint Path does NOT exist!")
    epochs = [int(f.split('_Epoch_')[-1].split('.')[0].replace('_checkpoint', '')) for f in files]
    epoch = max(epochs) if args.epoch == '-1' else args.epoch
    # print(files)
    # print(epochs)
    # print(epoch)
    bbone_path = os.path.join(args.ckpt_path, f"Backbone_Epoch_{epoch}_checkpoint.pth")
    noise_path = os.path.join(args.ckpt_path, f"Noise_Epoch_{epoch}_checkpoint.pth")

    if not os.path.exists(bbone_path):
        raise RuntimeError("Backbone Weights Not Found!")

    backbone = get_model('IR_50')(input_size, input_channel= 189)
    # backbone.load_state_dict(torch.load(bbone_path, map_location="cpu", weights_only= True))
    backbone.load_state_dict(torch.load(bbone_path, map_location="cpu"))
    backbone.eval()
    print("Backbone Loaded")

    noise_model = None
    if args.use_noise:
        noise_model = NoisyActivation(input_shape= 112, budget_mean= args.epsilon)
        # loading its weights
        if not os.path.exists(noise_path):
            print("Using random init for Noise weights")
        else:
            noise_model.load_state_dict(torch.load(noise_path, map_location="cpu"))
        noise_model.eval()
        print("Noise Model loaded")
    
    model = DCTDPModel(backbone, noise_model)
    model = model.to(device)
    model.eval()
    print("Model ready for evaulation")

    # load val data
    images, issame_list = get_val_pair_from_bin(args.data_root, args.bin_name)
    # run the validation
    print(len(issame_list), "Images loaded")
    acc, thresh = perform_val_bin(512, int(args.batch_size), model, images, issame_list)
    print(f"Model Name: {args.model_name} | Data Name: {args.data_name} | Accuracy: {acc * 100:.2f}% | Best Cosine Threshold: {thresh:.4f}")

if __name__ == '__main__':
    main()