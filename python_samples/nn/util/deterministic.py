import os
import random
import torch
import numpy as np

def set_deterministic(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python hash seed
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs

    # Force PyTorch to use deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # Disable the CUDA backend's non-deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32 - 1) + worker_id # Or any other deterministic way to derive worker seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def apply_deterministic(deterministic, seed):
    if deterministic:
        set_deterministic(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        loader_worker_init = seed_worker
    else:
        g = None
        loader_worker_init = None
    return g, loader_worker_init