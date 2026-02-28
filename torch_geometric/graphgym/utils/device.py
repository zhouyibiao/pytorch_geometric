import os
import subprocess

import numpy as np
import torch

from torch_geometric.graphgym.config import cfg


def get_gpu_memory_map():
    """Get the current GPU usage."""
    result = subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,nounits,noheader'
    ], encoding='utf-8')
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    return gpu_memory


def get_current_gpu_usage():
    """Get the current GPU memory usage."""
    if cfg.gpu_mem and cfg.device != 'cpu' and hasattr(torch, 'musa') and torch.musa.is_available():
        # TODO fix the GPU memory usage for MUSA, currently it is not accurate and may cause OOM error.
        result = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid,used_memory',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        current_pid = os.getpid()
        used_memory = 0
        for line in result.strip().split('\n'):
            line = line.split(', ')
            if current_pid == int(line[0]):
                used_memory += int(line[1])
        return used_memory
    else:
        return -1


def auto_select_device():
    r"""Auto select device for the current experiment."""
    if cfg.accelerator == 'auto':
        if hasattr(torch, 'musa') and torch.musa.is_available():
            cfg.accelerator = 'musa'
            cfg.devices = 1
        else:
            cfg.accelerator = 'cpu'
            cfg.devices = None
