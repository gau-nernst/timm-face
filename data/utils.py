import warnings

import torch
import torch.distributed as dist
import torchvision
from torch.utils.data import get_worker_info

# suppress PyTorch's complains
warnings.filterwarnings("ignore", message="The given buffer is not writable", category=UserWarning)


def decode_img(data: bytes):
    # before 0.20, only supports jpeg and png. after 0.20, additionally support webp.
    return torchvision.io.decode_image(torch.frombuffer(data, dtype=torch.uint8))


def _get_dist_info(*, include_worker_info: bool = False):
    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    worker_info = get_worker_info()
    if include_worker_info and worker_info is not None:
        rank = rank * worker_info.num_workers + worker_info.id
        world_size *= worker_info.num_workers
    return rank, world_size
