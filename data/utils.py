import warnings

import torch
import torchvision

# suppress PyTorch's complains
warnings.filterwarnings("ignore", message="The given buffer is not writable", category=UserWarning)


def decode_img(data: bytes):
    # before 0.20, only supports jpeg and png. after 0.20, additionally support webp.
    return torchvision.io.decode_image(torch.frombuffer(data, dtype=torch.uint8))
