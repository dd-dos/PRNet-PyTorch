import numpy as np
import scipy.io as sio
import torch
from PIL import Image
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os

from . import augmentation


class ImageData:
    def __init__(self):
        self.cropped_image_path = ''
        self.cropped_posmap_path = ''
        self.init_image_path = ''
        self.init_posmap_path = ''
        self.texture_path = ''
        self.texture_image_path = ''
        self.bbox_info_path = ''
        self.offset_posmap_path = ''
        self.attention_mask_path = ''

        self.image = None
        self.posmap = None
        self.offset_posmap = None
        self.bbox_info = None
        self.S = None
        self.T = None
        self.R = None
        self.attention_mask = None

    def readPath(self, image_dir):
        image_name = image_dir.split('/')[-1]
        self.cropped_image_path = image_dir + '/' + image_name + '_cropped.jpg'
        self.cropped_posmap_path = image_dir + '/' + image_name + '_cropped_uv_posmap.npy'
        self.init_image_path = image_dir + '/' + image_name + '_init.jpg'
        self.init_posmap_path = image_dir + '/' + image_name + '_uv_posmap.npy'
        # change the format to npy
        self.texture_path = image_dir + '/' + image_name + '_uv_texture_map.npy'
        self.texture_image_path = image_dir + '/' + image_name + '_uv_texture_map.jpg'

        self.bbox_info_path = image_dir + '/' + image_name + '_bbox_info.mat'

    def readFile(self, mode='posmap'):
        if mode == 'posmap':
            self.image = io.imread(self.cropped_image_path).astype(np.uint8)
            self.posmap = np.load(self.cropped_posmap_path).astype(np.float16)
        else:
            pass

    def getImage(self):
        if self.image is None:
            return io.imread(self.cropped_image_path)
        else:
            return self.image

    def getPosmap(self):
        if self.posmap is None:
            return np.load(self.cropped_posmap_path)
        else:
            return self.posmap

    def getOffsetPosmap(self):
        if self.offset_posmap is None:
            return np.load(self.offset_posmap_path)
        else:
            return self.offset_posmap

    def getBboxInfo(self):
        if self.bbox_info is None:
            return sio.loadmat(self.bbox_info_path)
        else:
            return self.bbox_info

    def getAttentionMask(self):
        if self.attention_mask is None:
            return np.load(self.attention_mask_path)
        else:
            return self.attention_mask


def toTensor(image):
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    return image

class DataGenerator(Dataset):
    def __init__(self, all_image_data, mode='posmap', is_aug=False, is_pre_read=True):
        super(DataGenerator, self).__init__()
        self.all_image_data = all_image_data
        self.image_height = 256
        self.image_width = 256
        self.image_channel = 3
        # mode=posmap or offset
        self.mode = mode
        self.is_aug = is_aug

        self.toTensor = transforms.ToTensor()

        self.is_pre_read = is_pre_read
        if is_pre_read:
            i = 0
            print('preloading')

            num_max_PR = 80000
            for data in self.all_image_data:
                data.readFile(mode=self.mode)
                print(i, end='\r')
                i += 1
                if i > num_max_PR:
                    break

    def __getitem__(self, index):
        if self.mode == 'posmap':
            image = (self.all_image_data[index].getImage() / 255.0).astype(np.float32)
            pos = self.all_image_data[index].getPosmap().astype(np.float32)
            # data augmentation
            if self.is_aug:
                image, pos = augmentation.prnAugment_torch(image, pos)
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            else:
                for i in range(3):
                    image[:, :, i] = (image[:, :, i] - image[:, :, i].mean()) / np.sqrt(image[:, :, i].var() + 0.001)
                image = self.toTensor(image)
            pos = pos / 280.
            pos = self.toTensor(pos)
            return image, pos
        else:
            import os
            os.error('please use "posmap" mode')
            return None

    def __len__(self):
        return len(self.all_image_data)


def getDataLoader(all_image_data, mode='posmap', batch_size=16, is_shuffle=False, is_aug=False, is_pre_read=True, num_worker=8):
    dataset = DataGenerator(all_image_data=all_image_data, mode=mode, is_aug=is_aug, is_pre_read=is_pre_read)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_worker, pin_memory=True)
    return train_loader


def FaceDataset(Dataset):
    def __init__(self, root, aug=True):
        self.root = root
        self.img_list, self.uv_list = self._get_data()
        self.aug = aug

    def __len__(self):
        assert len(self.img_list) == len(self.uv_list)
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        uv_path = self.img_list[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        uv =  np.load(uv_path)
        img, uv = self._preprocess(img, uv)

        return img, uv 

    def _get_data(self):
        img_list = []
        uv_list = []

        for root_dir, dirs, files in os.walk(self.root):
            for dir_name in dirs:
                image_name = dir_name
                if not os.path.exists(root + '/' + dir_name + '/' + image_name + '_cropped.jpg'):
                    print('skip ', root + '/' + dir_name)
                    continue
                img_path = root + '/' + dir_name + '/' + image_name + '_cropped.jpg'
                uv_path = root + '/' + dir_name + '/' + image_name + '_cropped_uv_posmap.npy'
                
                img_list.append(img_path)
                uv_list.append(img_path)

        return img_list, uv_list

    
    def _preprocess(self, img, uv):
        _img = (img/255.0).astype(np.float32)
        _uv = uv.astype(np.float32)

        if self.is_aug:
            _img, _uv = augmentation.prnAugment_torch(_img, _uv)
        for i in range(3):
                _img[:, :, i] = (_img[:, :, i] - _img[:, :, i].mean()) / np.sqrt(_img[:, :, i].var() + 0.001)
        _img = self.toTensor(_img)
        _uv = _uv / 280.
        _uv = self.toTensor(_uv)

        return _img, _uv