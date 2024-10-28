import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torchvision.transforms import v2

from .insightface import InsightFaceRecordIoDataset
from .utils import RepeatedSampler, ShuffleDataset, decode_img
from .webdataset import WebDataset


def cycle(dloader: DataLoader, device: str = "cpu"):
    epoch_idx = 0
    while True:
        if hasattr(dloader.sampler, "set_epoch"):
            dloader.sampler.set_epoch(epoch_idx)
        for batch in dloader:
            yield tuple(x.to(device) for x in batch)
        epoch_idx += 1


def create_train_dloader(
    path: str,
    batch_size: int,
    augmentations: list[str] | None = None,
    n_workers: int = 4,
    repeated_augmentation: int = 0,
    device: str = "cpu",
):
    augmentations = augmentations or []
    transform_list = [
        v2.ToImage(),
        *[eval(aug, dict(v2=v2)) for aug in augmentations],
    ]
    transform = v2.Compose(transform_list)

    if path.startswith("wds"):
        # standard webdataset
        if path.startswith("wds://"):
            import webdataset as wds

            ds = (
                wds.WebDataset(
                    path.removeprefix("wds://"),
                    shardshuffle=True,
                    nodesplitter=wds.split_by_node,
                )
                .shuffle(10_000, initial=10_000)
                .to_tuple("jpg", "cls")
                .map_tuple(lambda x: transform(decode_img(x)), lambda x: int(x.decode()))
            )

        # custom webdataset reader
        elif path.startswith("wds_hf://"):
            ds = WebDataset.from_hf(path.removeprefix("wds_hf://"), transform=transform)
            ds = ShuffleDataset(ds, buffer_size=10_000)

        elif path.startswith("wds_folder://"):
            ds = WebDataset.from_folder(path.removeprefix("wds_folder://"), transform=transform)
            ds = ShuffleDataset(ds, buffer_size=10_000)

        else:
            raise ValueError(f"Unsupport {path=}")

        assert repeated_augmentation == 0, "Not supported"
        dloader = DataLoader(ds, batch_size, num_workers=n_workers, pin_memory=True)
        ds_length = float("inf")

    else:
        ds = InsightFaceRecordIoDataset(path, transform=transform)

        if repeated_augmentation > 0:
            sampler = RepeatedSampler(ds, repeated_augmentation, shuffle=True)
        elif dist.is_initialized():
            sampler = DistributedSampler(ds, shuffle=True, drop_last=True)
        else:
            sampler = RandomSampler(ds)

        dloader = DataLoader(
            ds,
            batch_size,
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=True,
            drop_last=True,
        )
        ds_length = len(ds)

    return cycle(dloader, device=device), ds_length
