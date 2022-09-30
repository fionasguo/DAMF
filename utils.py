import random
import os
import psutil
import subprocess
import argparse
import logging
import numpy as np
import torch
from ast import literal_eval
import transformers


def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_gpu_memory_map():
    """
    Get the current gpu usage.
    returns a dict: key - device id int; val - memory usage in MB (int).
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return {'cpu_count': os.cpu_count(), '% RAM used': psutil.virtual_memory()[2]}

    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def count_devices():
    """
    Get the number of GPUs, if using CPU only, return 1
    """
    n_devices = torch.cuda.device_count()

    if n_devices == 0:
        n_devices = 1

    return n_devices

def read_args():
    # command args
    parser = argparse.ArgumentParser(description='Domain adaptation model for moral foundation inference.')
    parser.add_argument('-m','--mode',type=str,help='train,test,or train_test')
    parser.add_argument('-c','--config_path',type=str,help='configuration file dir that specifies hyperparameters etc')
    parser.add_argument('-i','--data_dir',type=str,help='input data directory')
    parser.add_argument('-o','--output_dir',type=str,help='output directory')
    parser.add_argument('-s','--seeds',type=str,help='list of seeds to evaluate')
    command_args = parser.parse_args()

    mode = command_args.mode

    # read args in config file
    args = {}
    with open(command_args.config_path,'r') as f:
        for l in f.readlines():
            if l.strip() == '' or l.strip().startswith('#'):
                continue
            arg = l.strip().split(" = ")
            arg_name,arg_val = arg[0],arg[1]
            if 'path' in arg_name:
                arg_val = os.path.join(os.path.dirname(os.path.realpath(__file__)),arg_val)
            args[arg_name] = arg_val

    if command_args.data_dir:
        args['data_dir'] = command_args.data_dir
    if command_args.output_dir:
        args['output_path'] = command_args.output_dir
    if command_args.seeds:
        args['seed_lst'] = literal_eval(command_args.seeds)
    if not os.path.exists(args['output_path']):
        os.makedirs(args['output_path'])
    args['domain_adapt'] = literal_eval(args['domain_adapt'])
    args['transformation'] = literal_eval(args['transformation'])
    args['reconstruction'] = literal_eval(args['reconstruction'])
    args['semi_supervised'] = literal_eval(args['semi_supervised'])
    if args['train_domain'][0] == '[':
        args['train_domain'] = literal_eval(args['train_domain'])
    else:
        args['train_domain'] = [args['train_domain']]
    if args['test_domain'][0] == '[':
        args['test_domain'] = literal_eval(args['test_domain'])
    else:
        args['test_domain'] = [args['test_domain']]
    args['n_mf_classes'] = int(args['n_mf_classes'])
    args['n_domain_classes'] = len(set(args['train_domain']+args['test_domain']))
    args['lr'] = float(args['lr'])
    args['alpha'] = float(args['alpha'])
    args['beta'] = float(args['beta'])
    args['batch_size'] = int(args['batch_size'])
    args['n_epoch'] = int(args['n_epoch'])
    args['dropout_rate'] = float(args['dropout_rate'])
    args['seed'] = int(args['seed'])
    args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        args['lambda_trans'] = float(args['lambda_trans'])
    except:
        pass
    try:
        args['lambda_rec'] = float(args['lambda_rec'])
    except:
        pass
    try:
        args['lambda_domain'] = 0.0
        args['num_no_adv'] = int(args['num_no_adv'])
        args['gamma'] = float(args['gamma'])
    except:
        pass

    logging.info('Configurations:')
    logging.info(args)

    return mode, args
