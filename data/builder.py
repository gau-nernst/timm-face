from torch.utils.data import DataLoader
from torchvision.transforms import v2

from .insightface import InsightFaceRecordIoDataset
from .webdataset import WebDataset


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

    if path.startswith("wds_"):
        if path.startswith("wds_hf://"):
            ds = WebDataset.from_hf(path.removeprefix("wds_hf://"), transform=transform)

        elif path.startswith("wds_folder://"):
            ds = WebDataset.from_folder(path.removeprefix("wds_folder://"), transform=transform)

        else:
            raise ValueError(f"Unsupport {path=}")

        dloader = DataLoader(ds, batch_size, num_workers=n_workers, pin_memory=True)
        ds_length = float("inf")

    else:
        ds = InsightFaceRecordIoDataset(path, transform=transform)
        dloader = DataLoader(ds, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True)
        ds_length = len(ds)

    return cycle(dloader, device=device), ds_length
