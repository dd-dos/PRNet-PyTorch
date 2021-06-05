import os

import cv2
import numpy as np
import torch
import random
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from utils.augmentation import rotateData

from . import augmentation
from .data import toTensor


class FaceDataset(Dataset):
    def __init__(self, root='', aug=True):
        super(Dataset).__init__()
        self.root = root
        self.img_list, self.uv_list = self._get_data()
        self.aug = aug

    def __len__(self):
        # return 4

        assert len(self.img_list) == len(self.uv_list)
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        uv_path = self.uv_list[idx]
        
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        uv =  np.load(uv_path, allow_pickle=True)
        import ipdb; ipdb.set_trace(context=10)

        img, uv, rotate_angle = self._preprocess(img, uv)

        meta = {
            'img_path': self.img_list[idx],
            'rotate_angle': rotate_angle
        }

        return img, uv, meta

    def _get_data(self):
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
        _img = (img/255.0).astype(np.float32)
        _uv = uv.astype(np.float32)

        rotate_angle = 0
        if self.aug:
            if np.random.rand() > 0.5:
                _img, _uv, rotate_angle = rotateData(_img, _uv, 90)
            _img, _uv = augmentation.prnAugment_torch(_img, _uv)
        
        for i in range(3):
                _img[:, :, i] = (_img[:, :, i] - _img[:, :, i].mean()) / np.sqrt(_img[:, :, i].var() + 0.001)
        
        _img = toTensor(_img)
        _uv = _uv / 280.
        _uv = toTensor(_uv)

        return _img, _uv, rotate_angle
