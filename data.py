import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


np.bool = bool  # fix for mxnet
from mxnet.recordio import MXIndexedRecordIO, unpack


class InsightFaceDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = Path(path)
        self.record = MXIndexedRecordIO(str(self.path / "train.idx"), str(self.path / "train.rec"), "r")

        header, _ = unpack(self.record.read_idx(0))
        self.size = int(header.label[0]) - 1
        self.n_classes = int(open(self.path / "property").read().split(",")[0])

        transform_list = [
            v2.ToImage(),
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

        img = Image.open(io.BytesIO(raw_img))
        img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return self.size
