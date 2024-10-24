import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .insightface import InsightFaceRecordIoDataset
from .utils import decode_img


def cycle(dloader: DataLoader, device: str = "cpu"):
    while True:
        for batch in dloader:
            yield tuple(x.to(device) for x in batch)


def create_train_dloader(
    path: str,
    batch_size: int,
    augmentations: list[str] | None = None,
    n_workers: int = 4,
    device: str = "cpu",
):
    augmentations = augmentations or []
    transform_list = [
        v2.ToImage(),
        *[eval(aug, dict(v2=v2)) for aug in augmentations],
    ]
    transform = v2.Compose(transform_list)

    if path.startswith("wds://"):
        import webdataset as wds

        path = path.removeprefix("wds://")
        ds = (
            wds.WebDataset(path, shardshuffle=True, nodesplitter=wds.split_by_node)
            .shuffle(10_000, initial=10_000)
            .to_tuple("jpg", "cls")
            .map_tuple(lambda x: transform(decode_img(x)), lambda x: int(x.decode()))
            .batched(batch_size, partial=False)
        )
        dloader = DataLoader(ds, None, num_workers=n_workers, pin_memory=True)
        ds_length = float("inf")

    else:
        ds = InsightFaceRecordIoDataset(path, transform=transform)
        dloader = DataLoader(ds, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True)
        ds_length = len(ds)

    return cycle(dloader, device=device), ds_length
