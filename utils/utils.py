# utils/utils.py
import random
import numpy as np
import torch
import gc
import os

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def memory_stats():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserv = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory | Allocated: {alloc:.2f}GB | Reserved: {reserv:.2f}GB | Total: {total:.2f}GB")
    else:
        print("Running on CPU")

