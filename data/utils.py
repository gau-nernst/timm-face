import warnings

import torch
import torch.distributed as dist
import torchvision
from torch.utils.data import Dataset, IterableDataset, Sampler, get_worker_info

# suppress PyTorch's complains
warnings.filterwarnings("ignore", message="The given buffer is not writable", category=UserWarning)


def decode_img(data: bytes):
    # before 0.20, only supports jpeg and png. after 0.20, additionally support webp.
    return torchvision.io.decode_image(torch.frombuffer(data, dtype=torch.uint8))


def _get_dist_info(*, include_worker_info: bool = False):
    """Return (rank, world_size)"""
    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank, world_size = 0, 1

    worker_info = get_worker_info()
    if include_worker_info and worker_info is not None:
        rank = rank * worker_info.num_workers + worker_info.id
        world_size *= worker_info.num_workers
    return rank, world_size


def get_rng(seed: int | None):
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    else:
        # seed from a true RNG source, then broadcast to all ranks
        rng.seed()
        if dist.is_initialized():
            state = rng.get_state()
            dist.broadcast(state, 0)
            rng.set_state(state)
    return rng


class ShuffleDataset(IterableDataset):
    def __init__(self, ds: IterableDataset, buffer_size: int = 1000, seed: int = 2024) -> None:
        self.ds = ds
        self.buffer_size = buffer_size

        self.rng = get_rng(seed)
        self.buffer1 = []
        self.buffer2 = []

    def __iter__(self):
        for sample in self.ds:
            # buffer2 is filled. once buffer2 is full, we shuffle and swap it with buffer1.
            # buffer2 should now be empty and buffer1 is full. we yield 1 sample from buffer1.
            # in subsequent iterations, we add 1 item to buffer2 and remove 1 item from buffer1,
            # thus maintaining the invariance that len(buffer1) + len(buffer2) = buffer_size - 1.
            self.buffer2.append(sample)
            if len(self.buffer2) == self.buffer_size:
                self.buffer2 = self._shuffle(self.buffer2)
                self.buffer1, self.buffer2 = self.buffer2, self.buffer1

            if len(self.buffer1):
                yield self.buffer1.pop()

        while len(self.buffer1):
            yield self.buffer1.pop()
        self.buffer2 = self._shuffle(self.buffer2)
        while len(self.buffer2):
            yield self.buffer2.pop()

    def _shuffle(self, buffer: list):
        indices = torch.randperm(len(buffer), generator=self.rng)
        return [buffer[idx] for idx in indices]


# https://github.com/facebookresearch/deit/blob/main/samplers.py
# first introduced in Batch Augmentation https://arxiv.org/abs/1901.09335
# also known as Repeated Augmentation in MultiGrain https://arxiv.org/abs/1902.05509
class RepeatedSampler(Sampler):
    def __init__(self, dataset: Dataset, num_repeats: int = 2, shuffle: bool = True) -> None:
        assert shuffle
        self.dataset = dataset
        self.num_repeats = num_repeats

        self.rng = get_rng()
        self.rank, self.world_size = _get_dist_info()
        self.size_per_rank = len(dataset) * num_repeats // self.world_size
        self.total_size = self.size_per_rank * self.world_size

    def __iter__(self):
        indices = torch.randperm(len(self.dataset), generator=self.rng)
        indices = torch.repeat_interleave(indices, self.num_repeats).tolist()
        indices = indices[self.rank :: self.world_size][: self.size_per_rank]
        assert len(indices) == self.size_per_rank
        return iter(indices)

    def __len__(self):
        return self.size_per_rank
