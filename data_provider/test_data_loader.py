import torch.utils.data as data
import os
import cv2

from utils import cfg
from data_provider import BaseTransform


class TestDataLoader(data.Dataset):
    def __init__(self, img_root):
        self.img_root = img_root
        self.transform = BaseTransform(sizes=cfg.input_size)
        self.img_list = sorted(os.listdir(self.img_root))

    def __getitem__(self, item):
        img_id = self.img_list[item]
        img = cv2.imread(os.path.join(self.img_root, img_id))
        img = self.transform([img])[0]
        img_id = int(img_id.split('.')[0]) # pytorch dataloader does not handle strings so use an int instead
        return img, img_id

    def __len__(self):
        return len(self.img_list)
