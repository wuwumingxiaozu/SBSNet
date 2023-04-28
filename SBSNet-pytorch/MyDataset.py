import os

import numpy as np
import torch
from until import *

from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path, imagesize=(1024, 1024)):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
        self.imagesize = imagesize

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'PNGImages', segment_name)
        segment_image = keep_image_size_open(segment_path, size=self.imagesize)
        image = keep_image_size_open_rgb(image_path, size=self.imagesize)
        return transform(image), torch.Tensor(np.array(segment_image))


if __name__ == '__main__':
    from torch.nn.functional import one_hot

    data = MyDataset('datasets')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out = one_hot(data[0][1].long())
    print(out.shape)
