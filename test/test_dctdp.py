# Author: Devashish Tripathi
# Based on codes originally licensed under Apache 2.0 by Tencent.


import os
import argparse
import glob
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchkit.backbone import get_model
from utils import perform_val_bin, get_val_pair_from_bin
from training.utils import NoisyActivation, images_to_batch

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
    parser.add_argument('--epoch', default= -1, type= int, help= 'Weights of which epoch. Select -1 for latest')
    parser.add_argument('--gpu_ids', default = '0', help= 'GPU IDs; comma separated')
    parser.add_argument('--data_root', default='', required=True, help='eval data root') 
    parser.add_argument('--bin_name', default='', required=True, help='name of bin file to eval. use conv script if needed.') 
    # parser.add_argument('--out_path', default='./output', help='output path')
    parser.add_argument('--model_name', default='test', help='name of model')
    parser.add_argument('--data_name', default='sample', help='name of eval dataset')
    parser.add_argument('--batch_size', default=64, type= int, help='batch size')
    parser.add_argument('--random_seed', default=1337, type= int, help='random seed')
    parser.add_argument('--epsilon', default=0.5, type= float, help='privacy budget')
    parser.add_argument('--no_noise', action='store_true' , help='toggle off noise adding module. for debugging')
    parser.add_argument('--no_locs', action='store_true', help='toggle off locs in NoisyActivation. Don\'t use this if you want as close as baseline')
    parser.add_argument('--use_sensitivity', action='store_true', help='Disable to follow the baseline (sensitivity=1)')
    parser.add_argument('--sens_pth', type=str, default=None, help='Path to dataset sensitivity tensor. Make using make_sense.py script!')


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
    
    # load val data
    images, issame_list = get_val_pair_from_bin(args.data_root, args.bin_name)
    # run the validation
    print(len(issame_list), "Image Pairs loaded")
    # print(len(images))
    # print(issame_list)
    # print(images)
    # return
    
    # load model
    if not os.path.exists(args.ckpt_path):
        raise RuntimeError("Checkpoint Path does NOT exist!")

    # {Backbone, Noise}_Epoch_X_checkpoint.pth
    path = os.path.join(args.ckpt_path, f"Backbone_Epoch_*_checkpoint.pth")
    files = glob.glob(path)
    if not files:
        raise RuntimeError("Checkpoint Path does NOT exist!")
    epochs = [int(f.split('_Epoch_')[-1].split('.')[0].replace('_checkpoint', '')) for f in files]
    epoch = max(epochs) if args.epoch == -1 else args.epoch
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
    if not args.no_noise:
        noise_model = NoisyActivation(input_shape= 112, budget_mean= float(args.epsilon), donot_use_loc= args.no_locs, 
                                      sensitivity= args.use_sensitivity, sens_pth= args.sens_pth)
        # loading its weights
        if not os.path.exists(noise_path):
            print("Using random init for Noise weights")
        else:
            noise_model.load_state_dict(torch.load(noise_path, map_location="cpu"))
        noise_model.eval()
        print("Noise Model loaded")

    # if noise_model:
    if not args.no_locs:
        print("locs mean/std:", noise_model.locs.mean().item(), noise_model.locs.std().item())
    print("rhos mean/std:", noise_model.rhos.mean().item(), noise_model.rhos.std().item())

    model = DCTDPModel(backbone, noise_model)
    model = model.to(device)
    model.eval()
    print("Model ready for evaluation")

    acc, thresh = perform_val_bin(512, int(args.batch_size), model, np.asarray(images), issame_list, tta= False)
    print(f"Model Name: {args.model_name} | Data Name: {args.data_name} | Accuracy: {acc * 100:.2f}% | Best Cosine Threshold: {thresh:.4f}")

if __name__ == '__main__':
    main()