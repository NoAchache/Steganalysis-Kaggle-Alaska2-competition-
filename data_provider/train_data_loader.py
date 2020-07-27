import torch.utils.data as data
import os
import cv2
import random

from utils import cfg


class TrainDataLoader(data.Dataset):
    def __init__(self, img_root, transform, is_training):
        self.is_training = is_training
        self.cover_root = os.path.join(img_root, 'Cover')
        self.noisy_root = [os.path.join(img_root, dir_name) for dir_name in ['JUNIWARD', 'JMiPOD', 'UERD']]
        self.transform = transform
        img_list = sorted(os.listdir(self.cover_root))
        random.Random(42).shuffle(img_list) #Shuffle dataset for better homogeneousity. Use seed of 42 for repeatability
        num_img_train = int(len(img_list) * cfg.train_data_share)
        self.img_list = img_list[:num_img_train] if self.is_training else img_list[num_img_train:]

    def __getitem__(self, item):
        image_id = self.img_list[item]
        cover_img = cv2.imread(os.path.join(self.cover_root, image_id))
        noisy_img = [cv2.imread(os.path.join(noisy_path, image_id)) for noisy_path in self.noisy_root]
        preprocessed_imgs = self.transform([cover_img] + noisy_img)
        if self.is_training:
            return preprocessed_imgs #no needs to return labels since they are created later on
        else:
            labels = [0, 1, 2, 3]
            return preprocessed_imgs, labels

    def __len__(self):
        return len(self.img_list)
