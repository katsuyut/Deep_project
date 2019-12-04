import os, sys, time
import shutil
import numpy as np
import argparse
import chainer
from PIL import Image

base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base, '../'))
from evaluation import gen_images_with_condition
import yaml
import source.yaml_utils as yaml_utils


def load_models(config):
    gen_conf = config.models['generator']
    gen = yaml_utils.load_model(gen_conf['fn'], gen_conf['name'], gen_conf['args'])
    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/base.yml')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--results_dir', type=str, default='./results/gans')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--batchsize', type=int, default=26)
    parser.add_argument('--num', type=int, default=10000)
    args = parser.parse_args()
    chainer.cuda.get_device_from_id(args.gpu).use()
    config = yaml_utils.Config(yaml.load(open(args.config_path)))
    # Model
    gen = load_models(config)
    gen.to_gpu(args.gpu)
    out = args.results_dir
    chainer.serializers.load_npz(args.snapshot, gen)
    np.random.seed(1234)
    for i in range(3):
      for c in range(129):
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            x = gen_images_with_condition(gen, c=c, n=args.batchsize, batchsize=args.batchsize)
        _, _, h, w = x.shape
        x = x.reshape((args.batchsize, 3, h, w))
        
        for b in range(args.batchsize):
          x_save = x[b].transpose(1,2,0)
          save_path = os.path.join(out, '{}.png'.format(str(i*129*26 + c*26 + b)))
          if not os.path.exists(out):
              os.makedirs(out)
          Image.fromarray(x_save).save(save_path)


if __name__ == '__main__':
    main()
