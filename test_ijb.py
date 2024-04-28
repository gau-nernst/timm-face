# references:
# https://github.com/mk-minchul/AdaFace/blob/master/validation_mixed/validate_IJB_BC.py
# https://github.com/deepinsight/insightface/blob/master/recognition/_evaluation_/ijb/ijb_11.py

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import safetensors
import sklearn.preprocessing
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm


class IJBDataset(Dataset):
    ARCFACE_KEYPOINTS = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )

    def __init__(self, ijb_dir: str, meta_dir: str, dataset: str) -> None:
        """
        ijb_dir: https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view
        meta_dir: https://drive.google.com/file/d/1MXzrU_zUESSx_242pRUnVvW_wDzfU8Ky/view
        """
        super().__init__()
        assert dataset in ("IJBB", "IJBC")
        self.img_dir = Path(ijb_dir) / dataset / "loose_crop"

        meta_path = Path(meta_dir) / f"{dataset}_meta/{dataset.lower()}_name_5pts_score.txt"
        self.keypoints_list = np.loadtxt(meta_path, dtype=np.float32, usecols=range(1, 11)).reshape(-1, 5, 2)

        transform_list = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = v2.Compose(transform_list)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_dir / f"{idx+1}.jpg").convert("RGB")
        kpts = self.keypoints_list[idx]

        # set ransacReprojThreshold=float("inf") so that all points are inliers
        M, _ = cv2.estimateAffinePartial2D(kpts, self.ARCFACE_KEYPOINTS, ransacReprojThreshold=float("inf"))
        img = cv2.warpAffine(np.array(img), M, (112, 112), flags=cv2.INTER_CUBIC)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.keypoints_list.shape[0]


def image2template_feature(img_embs: np.ndarray, template_ids: np.ndarray, media_ids: np.ndarray):
    unique_templates = np.unique(template_ids)
    template_feats = np.zeros((len(unique_templates), img_embs.shape[1]))

    for count_template, uqt in enumerate(tqdm(unique_templates, desc="Get template embeddings", dynamic_ncols=True)):
        (ind_t,) = np.where(template_ids == uqt)
        face_norm_feats = img_embs[ind_t]
        face_medias = media_ids[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            ind_m = face_medias == u
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)

    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    return template_norm_feats, unique_templates


def score_templates(template_embs: np.ndarray, unique_templates: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    template2id = np.empty(max(unique_templates) + 1, dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    p1 = template2id[p1]
    p2 = template2id[p2]

    scores = np.empty(len(p1), dtype=np.float32)
    batch_size = 100_000
    for start_idx in tqdm(range(0, len(p1), batch_size), desc="Get scores", dynamic_ncols=True):
        index = slice(start_idx, start_idx + batch_size)
        feat1 = template_embs[p1[index]]
        feat2 = template_embs[p2[index]]
        scores[index] = (feat1 * feat2).sum(-1)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ijb_dir", default="ijb")
    parser.add_argument("--meta_dir", default="meta")
    parser.add_argument("--dataset", required=True, choices=["IJBB", "IJBC"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_kwargs", type=json.loads)
    parser.add_argument("--ckpt")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp_dtype", default="none", choices=["none", "float16", "bfloat16"])
    args = parser.parse_args()

    model = timm.create_model(args.model, **(args.model_kwargs or dict())).eval().cuda()
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
    if args.compile:
        model.compile()

    if args.ckpt is not None:
        if args.ckpt.endswith(".safetensors"):
            with safetensors.safe_open(args.ckpt, framework="pt") as f:
                state_dict = {k: f.get_tensor(k) for k in f.keys()}
        else:
            state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict)

    amp_dtype = dict(none=None, float16=torch.float16, bfloat16=torch.bfloat16)[args.amp_dtype]

    ds = IJBDataset(args.ijb_dir, args.meta_dir, args.dataset)
    dloader = DataLoader(ds, args.batch_size, num_workers=args.n_workers, pin_memory=True)
    img_embs = np.empty((len(ds), 512), dtype=np.float32)
    img_idx = 0

    for images in tqdm(dloader, desc="Get image embeddings", dynamic_ncols=True):
        images = images.cuda()
        if args.channels_last:
            images = images.to(memory_format=torch.channels_last)

        with torch.no_grad(), torch.autocast("cuda", amp_dtype, amp_dtype is not None):
            embs = model(images)
        embs = F.normalize(embs.float(), dim=1).cpu().numpy()
        img_embs[img_idx : img_idx + embs.shape[0]] = embs
        img_idx += embs.shape[0]

    meta_path = f"{args.meta_dir}/{args.dataset}_meta/{args.dataset.lower()}_face_tid_mid.txt"
    template_ids, media_ids = np.loadtxt(meta_path, dtype=int, usecols=(1, 2), unpack=True)
    template_embs, unique_templates = image2template_feature(img_embs, template_ids, media_ids)

    meta_path = f"{args.meta_dir}/{args.dataset}_meta/{args.dataset.lower()}_template_pair_label.txt"
    p1, p2, labels = np.loadtxt(meta_path, dtype=int, unpack=True)
    scores = score_templates(template_embs, unique_templates, p1, p2)

    fpr, tpr, _ = roc_curve(labels, scores)
    far_targets = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    tpr_fpr_row = []
    for far_target in far_targets:
        idx = np.argmin(np.abs(fpr - far_target))
        tpr_fpr_row.append(tpr[idx])

    print(far_targets)
    print(tpr_fpr_row)
