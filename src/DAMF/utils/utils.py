"""Utility functions"""

import random
import os
import psutil
import subprocess
import logging
import numpy as np
import torch
from ast import literal_eval
import transformers


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_gpu_memory_map():
    """
    Get the current gpu usage.
    returns a dict: key - device id int; val - memory usage in GB (int).
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return {
            'cpu_count': os.cpu_count(),
            '% RAM used': psutil.virtual_memory()[2]
        }
        
    gpu_memory_map = {}
    for i in range(count_devices()):
        gpu_memory_map[i] = round(torch.cuda.memory_allocated(i)/1024/1024/1024,2)
    return gpu_memory_map


def count_devices():
    """
    Get the number of GPUs, if using CPU only, return 1
    """
    n_devices = torch.cuda.device_count()

    if n_devices == 0:
        n_devices = 1

    return n_devices


def read_config(curr_dir, args):
    """
    Read arguments from the config file.

    Args:
        curr_dir: the directory where the main script is being run
        args: a dict to store arguments, should at least include 'config_dir'
    """
    # read args in config file
    with open(args['config_dir'], 'r') as f:
        for l in f.readlines():
            # skip comments
            if l.strip() == '' or l.strip().startswith('#'):
                continue
            # get value
            arg = l.strip().split(" = ")
            arg_name, arg_val = arg[0], arg[1]

            args[arg_name] = arg_val

    ## format each arg
    # pretrained model dir
    args['pretrained_dir'] = os.path.join(curr_dir, args['pretrained_dir'])
    # booleans
    args['domain_adapt'] = literal_eval(args['domain_adapt'])
    args['transformation'] = literal_eval(args['transformation'])
    args['reconstruction'] = literal_eval(args['reconstruction'])
    args['semi_supervised'] = literal_eval(args['semi_supervised'])
    args['weighted_loss'] = literal_eval(args['weighted_loss'])
    # train/test domains
    if args['train_domain'][0] == '[':
        args['train_domain'] = literal_eval(args['train_domain'])
    else:
        args['train_domain'] = [args['train_domain']]
    if args['test_domain'][0] == '[':
        args['test_domain'] = literal_eval(args['test_domain'])
    else:
        args['test_domain'] = [args['test_domain']]
    # number of mf and domain classes
    args['n_mf_classes'] = int(args['n_mf_classes'])
    args['n_domain_classes'] = len(
        set(args['train_domain'] + args['test_domain']))
    # hyperparameters
    args['lr'] = float(args['lr'])
    args['alpha'] = float(args['alpha'])
    args['beta'] = float(args['beta'])
    args['batch_size'] = int(args['batch_size'])
    args['n_epoch'] = int(args['n_epoch'])
    args['dropout_rate'] = float(args['dropout_rate'])
    try:
        args['lambda_trans'] = float(args['lambda_trans'])
    except:
        args['lambda_trans'] = 0.0
    if args['lambda_trans'] == 0:
        args['transformation'] = False
    try:
        args['lambda_rec'] = float(args['lambda_rec'])
    except:
        args['lambda_rec'] = 0.0
    if args['lambda_rec'] == 0:
        args['reconstruction'] = False
    try:
        args['lambda_domain'] = 0.0
        args['num_no_adv'] = int(args['num_no_adv'])
        args['gamma'] = float(args['gamma'])
    except:
        args['lambda_domain'] = 0.0
        args['num_no_adv'] = 0
        args['gamma'] = 0.0
    # others
    args['seed'] = int(args['seed'])
    args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"

    logging.info('Configurations:')
    logging.info(args)

    return args
