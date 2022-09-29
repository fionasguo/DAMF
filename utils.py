import random
import os
import psutil
import numpy as np
import torch
import transformers
import subprocess

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
