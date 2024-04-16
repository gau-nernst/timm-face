import pickle
import struct
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torchvision as tv
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


# drop-in replacement for mxnet.recordio.MXIndexedRecordIO
class MXIndexedRecordIO:
    def __init__(self, idx_path: str, rec_path: str, mode: str) -> None:
        assert mode == "r"
        index = np.loadtxt(idx_path, dtype=int)
        self.offsets = torch.empty(index.shape[0], dtype=int)
        for idx, offset in index:
            self.offsets[idx] = offset
        self.rec = np.memmap(rec_path, dtype=np.uint8, mode="r")

    def read_idx(self, idx: int):
        offset = self.offsets[idx].item()
        length = self.rec[offset + 4 : offset + 8].view(np.uint32)[0]
        return self.rec[offset + 8 : offset + 8 + length]


class Header(NamedTuple):
    flag: int
    label: float | np.ndarray
    id: int
    id2: int


def unpack(s):
    flag, label, id, id2 = struct.unpack("IfQQ", s[:24])
    s = s[24:]
    if flag > 0:
        label = s[: flag * 4].view(np.float32)
        s = s[flag * 4 :]
    return Header(flag, label, id, id2), s


def decode_image_pt(data):
    return tv.io.decode_image(torch.frombuffer(data, dtype=torch.uint8))


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
    ds = InsightFaceRecordIoDataset(path, augmentations=augmentations)
    dloader = DataLoader(ds, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True)
    n_classes = int(open(Path(path) / "property").read().split(",")[0])
    return cycle(dloader, device=device), len(ds), n_classes


class InsightFaceRecordIoDataset(Dataset):
    def __init__(self, path: str, augmentations: list[str] | None = None):
        super().__init__()
        self.path = Path(path)
        self.record = MXIndexedRecordIO(str(self.path / "train.idx"), str(self.path / "train.rec"), "r")

        header, _ = unpack(self.record.read_idx(0))
        self.size = int(header.label[0]) - 1

        augmentations = augmentations or []
        transform_list = [
            v2.ToImage(),
            v2.RandomHorizontalFlip(),
            *[eval(aug, dict(v2=v2)) for aug in augmentations],
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = v2.Compose(transform_list)

    def __getitem__(self, idx: int):
        header, raw_img = unpack(self.record.read_idx(idx + 1))

        label = header.label
        if not isinstance(label, (int, float)):
            label = label[0]
        label = int(label)

        img = self.transform(decode_image_pt(raw_img))
        return img, label

    def __len__(self) -> int:
        return self.size


class InsightFaceBinDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.raw_images, self.labels = pickle.load(open(path, "rb"), encoding="bytes")

        transform_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = v2.Compose(transform_list)

    def __getitem__(self, idx: int):
        img1 = self.transform(decode_image_pt(self.raw_images[2 * idx]))
        img2 = self.transform(decode_image_pt(self.raw_images[2 * idx + 1]))
        label = int(self.labels[idx])
        return img1, img2, label

    def __len__(self) -> int:
        return len(self.labels)
