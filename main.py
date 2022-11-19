import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners import *

import os

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default = "cifar10.yml",  help='Path to the config file')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--exp', type=str, default='./exp', help='Path for saving running related data.')
    parser.add_argument('--model_dir', type=str, default="cifar10", help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('--fast_fid', action='store_true', help='Whether to do fast fid test')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='saved_results', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', default= True, help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--gpu', type = int, default= 0, help="cuda id")
    parser.add_argument('--kappa_id', type=int, default= 1, help="No interaction. Suitable for Slurm Job launcher")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.model_dir)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    if args.sample:
        os.makedirs(os.path.join(args.exp, args.image_folder), exist_ok=True)
        quantized_bits = str(new_config.measurements.quantize_bits) + '_bit'
        if not new_config.measurements.quantization:
            quantized_bits = 'linear' 
        args.image_folder = os.path.join(args.exp, args.image_folder, new_config.data.dataset, quantized_bits)
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)

    # Device setting
    device_str =  f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    try:
        runner = NCSNRunner(args, config)
        if args.sample:
            runner.sample()
        else:
            raise ValueError('Only sampling is supported!')
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
