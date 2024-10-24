import logging
import tarfile
from pathlib import Path
from typing import Callable

import huggingface_hub
import requests
import requests.adapters
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from .utils import _get_dist_info, decode_img

logger = logging.getLogger(__name__)


class WebDataset(IterableDataset):
    def __init__(
        self,
        shards: list[str],
        img_key: str = "jpg",
        label_key: str = "cls",
        transform: Callable[[Tensor], Tensor] | None = None,
        eval: bool = True,
        seed: int = 2024,
    ) -> None:
        self.shards = shards
        self.img_key = img_key
        self.label_key = label_key
        self.transform = transform
        self.eval = eval

        self._rng = torch.Generator().manual_seed(seed)
        self._sess = None

    @staticmethod
    def from_hf(repo_id: str, **kwargs) -> "WebDataset":
        fs = huggingface_hub.HfFileSystem()
        urls = []
        for path in fs.glob(f"hf://datasets/{repo_id}/**/*.tar"):
            hf_file = fs.resolve_path(path)
            url = huggingface_hub.hf_hub_url(repo_id, hf_file.path_in_repo, repo_type="dataset")
            urls.append(url)
        urls.sort()
        return WebDataset(urls, **kwargs)

    @staticmethod
    def from_folder(data_dir: str, **kwargs) -> "WebDataset":
        shards = list(Path(data_dir).glob("**/*.tar"))
        shards.sort()
        return WebDataset(shards, **kwargs)

    def _open_url(self, url: str):
        if self._sess is None:
            self._sess = requests.Session()
            retries = requests.adapters.Retry(total=5, backoff_factor=0.1)
            http = requests.adapters.HTTPAdapter(max_retries=retries)
            self._sess.mount("http://", http)
            self._sess.mount("https://", http)

        headers = dict()
        if url.startswith("https://huggingface.co/datasets"):
            token = huggingface_hub.utils.get_token()
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"

        # TODO: might use smaller timeout. add retry for timeout/broken connection.
        # TODO: support local wds
        resp = self._sess.get(
            url,
            headers=headers,
            timeout=30,
            stream=True,
        )
        return tarfile.open(fileobj=resp.raw, mode="r|")

    def _open(self, shard: str):
        if Path(shard).exists():
            return tarfile.open(shard)
        else:
            return self._open_url(shard)

    def _shard_iter(self):
        while True:
            if not self.eval:
                indices = torch.randperm(len(self.shards), generator=self._rng)
            else:
                indices = range(len(self.shards))

            for idx in indices:
                yield self.shards[idx]
            if self.eval:
                break

    def __iter__(self):
        rank, world_size = _get_dist_info(include_worker_info=True)

        # if all distributed processes use the same RNG seed, the infinite url sequence should be exactly identical.
        # thus, to evenly distribute them among processes, each process simply takes 1 shard every world_size.
        for shard_idx, shard in enumerate(self._shard_iter()):
            if shard_idx % world_size != rank:
                continue

            try:
                tar = self._open(shard)
                key = img = label = None
                for tarinfo in tar:
                    _key, ext = tarinfo.name.rsplit(".", 1)

                    if key is not None:
                        if _key != key:
                            yield img, label
                            key = _key
                    else:
                        key = _key

                    if ext in (self.img_key, self.label_key):
                        data = tar.extractfile(tarinfo).read()

                        if ext == self.img_key:
                            img = decode_img(data)
                            if self.transform is not None:
                                img = self.transform(img)

                        elif ext == self.label_key:
                            label = int(data)

                yield img, label

            except Exception as e:
                # when failure, we simply continue with the next shard
                # TODO: might want to retry/resume with Range header instead
                logger.exception(f"Exception while reading {shard=}. {e}")
