from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from model import Glow

## added by Matz for visualization
from visualization import Visualizations
vis = Visualizations()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--batch', default=8, type=int, help='batch size')
parser.add_argument(
    '--n_flow', default=32, type=int, help='number of flows in each block'
)
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument(
    '--no_lu',
    action='store_true',
    help='use plain convolution instead of LU decomposed version',
)
parser.add_argument(
    '--affine', action='store_true', help='use affine coupling instead of additive'
)
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=10000, type=int, help='number of samples')
parser.add_argument('--ckpt_model', default="", type=str, help='Path ')


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def process(args, model):
    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    
    eval_batch = 100
    with torch.no_grad():
      for k in range(0, args.n_sample, eval_batch):
        print(k)
        z_sample = []
        for z in z_shapes:
          z_new = torch.randn(eval_batch, *z) * args.temp
          z_sample.append(z_new.to(device))
        tensor1 = model_single.reverse(z_sample).to(device).data
        
        # normalize
        normalize_range=(-0.5, 0.5)
        def norm_ip(img, norm_min, norm_max):
            img.clamp_(min=norm_min, max=norm_max)
            img.add_(-norm_min).div_(norm_max - norm_min + 1e-5)
            return img
        def norm_range(t, n_range):
            if n_range is not None:
                return norm_ip(t, n_range[0], n_range[1])
            else:
                return norm_ip(t, float(t.min()), float(t.max()))
        tensor2 = norm_range(tensor1, normalize_range)
        # end normalize
        
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = tensor2.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        
        for i in range(eval_batch):
          im = Image.fromarray(ndarr[i].transpose(1,2,0))
          im.save(f'out/{str(k + i).zfill(5)}.png')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    model = nn.DataParallel(model_single)
    model.load_state_dict(torch.load(args.ckpt_model))
    model = model.to(device)

    process(args, model)
