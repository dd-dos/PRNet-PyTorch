import os

import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from . import augmentation
from .data import toTensor

class FaceDataset(Dataset):
    def __init__(self, root='', aug=True):
        super(Dataset).__init__()
        self.root = root
        self.img_list, self.uv_list = self._load_data()
        self.aug = aug

    def __len__(self):
        assert len(self.img_list) == len(self.uv_list)
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        uv_path = self.uv_list[idx]
        
        img = cv2.imread(img_path)
        uv =  np.load(uv_path, allow_pickle=True).astype(np.float32)

        img, uv, rotate_angle, pre_normalized_img = self._preprocess(img, uv)

        meta = {
            'img_path': self.img_list[idx],
            'rotate_angle': rotate_angle,
            'pre_normalized_img': pre_normalized_img
        }

        return img, uv, meta

    def _load_data(self):
        img_list = []
        uv_list = []
        
        _root = Path(self.root)
        for img_path in _root.glob("**/*_cropped.jpg"):
            split_path = str(img_path).split("/")
            
            true_name = split_path[-2]
            parent_dir = "/".join(split_path[:-1])
            uv_path = os.path.join(parent_dir, true_name + '_cropped_uv_posmap.npy')

            img_list.append(str(img_path))
            uv_list.append(str(uv_path))

        return img_list, uv_list

    def _preprocess(self, img, uv):
        if self.aug:
            img, uv, rotate_angle = augmentation.prnAugment_torch(img, uv)

        # Save a copy for visualization
        pre_normalized_img = img.copy()

        # Convert to 0-1 scale 
        img, uv = self._normalize(img, uv)

        uv = toTensor(uv)
        img = toTensor(img)

        return img, uv, rotate_angle, np.array(pre_normalized_img)
    
    def _normalize(self, img, uv):
        img = (img/255.0).astype(np.float32)
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / np.sqrt(img[:, :, i].var() + 0.001)

        uv = uv / 280.

        return img, uv

    