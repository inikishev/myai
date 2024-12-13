import csv
import itertools
import os

import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from ..data import DS
from ..loaders.image import imreadtensor
from ..torch_tools import pad_to_shape
from .base import Dataset

# WRONG
MEAN = torch.tensor([[
    [[107.6875]],
    [[108.5625]],
    [[110.3750]]
]])
STD = torch.tensor([[
    [[74.8125]],
    [[75.3750]],
    [[76.9375]]
]])

_resize = v2.Resize((240, 320), antialias=False)
def _preprocess(x):
    if x.shape[0] == 1: x = torch.cat([x,x,x])
    return _resize(pad_to_shape(x,(3, 480, 640)),)

def _image_to_float32(x):
    image, label = x
    return image.to(torch.float32, copy=False), label

class _MISISCar(Dataset):

    root = r"E:\datasets\MDS-MISIS-DL Car classification"

    def _make(self):
        with open(os.path.join(self.root, "train.csv"), 'r', encoding = 'utf8') as f:
            items = list(csv.reader(f))[1:]

        # have to use float16 and resize by 50% to make it fit into memory
        ds = [(
            _preprocess(
                imreadtensor(
                    os.path.join(self.root, "train", f"{i[1]}", f"{i[0]}"),
                    dtype=torch.float16),
                ),
            int(i[1])
        ) for i in items]

        # stack
        images = torch.stack([i[0] for i in ds])
        labels = torch.tensor([i[1] for i in ds], dtype=torch.float32)

        # znorm
        images -= MEAN
        images /= STD

        return images, labels


    def _save(self):
        images, labels = self._make()
        np.savez_compressed(os.path.join(self.root, 'train 240x320.npz'), images=images, labels = labels)

    def get(self) -> DS[tuple[torch.Tensor, int]]:
        data = np.load(os.path.join(self.root, 'train 240x320.npz'))
        images = data['images']
        labels = data['labels']
        ds = DS()
        ds.add_samples_([(torch.from_numpy(i),
                          torch.tensor(l, dtype=torch.float32)) for i, l in zip(images, labels)],
                        loader=_image_to_float32)
        return ds


    def make_val_submission(self, outfile, model, batch_size = 32):
        submission = "Id,Category\n"
        for ids in itertools.batched(tqdm(os.listdir(os.path.join(self.root, "test"))), batch_size):
            inputs = torch.stack([_preprocess(imreadtensor(os.path.join(self.root, "test", f), dtype=torch.float32)) for f in ids])
            inputs -= MEAN
            inputs /= STD

            outputs = model(inputs)
            submission += '\n'.join([f'{i},{int(o)}' for i, o in zip(ids, outputs.argmax(1).detach().cpu())])
            submission += '\n'

        with open(outfile, 'w', encoding = 'utf8') as f:
            f.write(submission[:-1])

MISISCar = _MISISCar()
"""15561 images, original was 3×480×640 but I resized to 3×240×320 and saved in float16 so that it fits into memory.
entire dataset is znormalized per channel.
10 classes, each image has integer label 0 to 9. LOADER IS SET TO CAST TO FLOAT32"""
