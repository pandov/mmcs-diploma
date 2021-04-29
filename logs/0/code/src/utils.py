from typing import Dict

import torch

def byte2gb(x: float) -> float:
    return round(x / (1024 ** 3), 2)

def get_gpu_memory_info(n: int = 0) -> Dict[str, float]:
    total = torch.cuda.get_device_properties(n).total_memory
    reserved = torch.cuda.memory_reserved(n)
    allocated = torch.cuda.memory_allocated(n)
    free = total - allocated - reserved
    return {
        'total': byte2gb(total),
        'reserved': byte2gb(reserved),
        'allocated': byte2gb(allocated),
        'free': byte2gb(free),
    }
