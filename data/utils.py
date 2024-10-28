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


def sync_rng_state(rng: torch.Generator):
    if dist.is_initialized():
        state = rng.get_state()
        dist.broadcast(state, 0)
        rng.set_state(state)


class ShuffleDataset(IterableDataset):
    def __init__(self, ds: IterableDataset, buffer_size: int = 1000, seed: int = 2024) -> None:
        self.ds = ds
        self.buffer_size = buffer_size

        self._generator = torch.Generator().manual_seed(seed)
        self._buffer1 = []
        self._buffer2 = []

    def __iter__(self):
        for sample in self.ds:
            # buffer2 is filled. once buffer2 is full, we shuffle and swap it with buffer1.
            # buffer2 should now be empty and buffer1 is full. we yield 1 sample from buffer1.
            # in subsequent iterations, we add 1 item to buffer2 and remove 1 item from buffer1,
            # thus maintaining the invariance that len(buffer1) + len(buffer2) = buffer_size - 1.
            self._buffer2.append(sample)
            if len(self._buffer2) == self.buffer_size:
                self._buffer2 = self._shuffle(self._buffer2)
                self._buffer1, self._buffer2 = self._buffer2, self._buffer1

            if len(self._buffer1):
                yield self._buffer1.pop()

        while len(self._buffer1):
            yield self._buffer1.pop()
        self._buffer2 = self._shuffle(self._buffer2)
        while len(self._buffer2):
            yield self._buffer2.pop()

    def _shuffle(self, buffer: list):
        indices = torch.randperm(len(buffer), generator=self._generator)
        return [buffer[idx] for idx in indices]


# NOTE: support resume correctly
# https://github.com/facebookresearch/deit/blob/main/samplers.py
# first introduced in Batch Augmentation https://arxiv.org/abs/1901.09335
# also known as Repeated Augmentation in MultiGrain https://arxiv.org/abs/1902.05509
class RepeatedSampler(Sampler):
    def __init__(self, dataset: Dataset, num_repeats: int = 2, shuffle: bool = True) -> None:
        assert shuffle
        self.dataset = dataset
        self.num_repeats = num_repeats

        self.rank, self.world_size = _get_dist_info()
        self.epoch = 0
        self.size_per_rank = len(dataset) * num_repeats // self.world_size
        self.total_size = self.size_per_rank * self.world_size

    def __iter__(self):
        # deterministic sequence of data based on epoch idx
        indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.epoch))
        indices = torch.repeat_interleave(indices, self.num_repeats).tolist()
        indices = indices[self.rank :: self.world_size][: self.size_per_rank]
        assert len(indices) == self.size_per_rank
        return iter(indices)

    def __len__(self):
        return self.size_per_rank

    def set_epoch(self, epoch: int):
        self.epoch = epoch
