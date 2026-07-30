"""Super-resolution dataset (LR input / HR target pairs).

This module was missing from the repository even though ``train.py`` imports
``from src.data.dataset import LoadDataset``; it is reconstructed here to match
that call site (``LoadDataset(img_dir, lr_size, max_cache_size)``).

Convention
----------
The reconstruction models (``TransConv`` / ``UDUCNN`` / ``UUDCNN``) upsample the
spatial resolution by exactly 2x. So for a low-resolution input of side
``lr_size`` the network output — and therefore the training target — has side
``2 * lr_size``. Each sample is built from a high-resolution source image:

* ``target`` : the source resized (bicubic) to ``2 * lr_size`` (the HR image),
* ``input``  : ``target`` bicubic-downsampled to ``lr_size`` (the LR image).

Both are returned as float32 tensors in ``[0, 1]`` with shape ``(3, H, W)``.
"""

import glob
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode


class LoadDataset(Dataset):
    def __init__(self, img_dir, lr_size=256, max_cache_size=1000):
        self.img_dir = img_dir
        self.lr_size = int(lr_size)
        self.hr_size = 2 * int(lr_size)
        self.max_cache_size = int(max_cache_size)
        self._cache = {}

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(img_dir, ext)))
        self.paths = sorted(paths)

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path):
        if path in self._cache:
            return self._cache[path]
        img = Image.open(path).convert("RGB")
        if len(self._cache) < self.max_cache_size:
            self._cache[path] = img
        return img

    def __getitem__(self, idx):
        img = self._load_image(self.paths[idx])
        hr = resize(img, [self.hr_size, self.hr_size],
                    interpolation=InterpolationMode.BICUBIC)
        lr = resize(hr, [self.lr_size, self.lr_size],
                    interpolation=InterpolationMode.BICUBIC)
        return to_tensor(lr), to_tensor(hr)
