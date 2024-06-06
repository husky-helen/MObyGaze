import numpy as np
import torch
import os
import random

def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch


def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_tensor_list(tensor_list):
    """Load all the tensor according to a list of paths

    Args:
        tensor_list (list): List storing all the paths to the tensors to load (all the tensors should have a dimension in commom)

    Returns:
        torch.tensor: concatenated tensor of all the tensors that have been loaded
    """
  
    if len(tensor_list) < 2: return torch.load(tensor_list[0])[0]

    res = [torch.load(t) for t in tensor_list]
    res = torch.concat(res, dim = 0)
    return res


