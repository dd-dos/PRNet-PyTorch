import argparse
import logging
import os
import time

import cv2
import models
import numpy as np
import torch
from models.resfcn256 import ResFCN256
from utils.data import custom_crop, getLandmark, toTensor
from utils.visualize import demoKpt

logging.getLogger().setLevel(logging.INFO)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class FacePatternModel: 
    def __init__(self, model_path):
        '''
        Main class for extracting face pattern.

        :model_path: path to model weights. Currently support PRnet.
        '''
        self.model = ResFCN256()
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    @torch.no_grad()
    def draw_landmarks(self, img, bboxes):
        cropped_faces = []
        height, width, _ = img.shape
        offset = []

        for bbox in bboxes:
            if bbox[-1] >= 0.9:
                length = max(bbox[2]-bbox[0], bbox[3]-bbox[1])*1.2
                center = [bbox[0]+(bbox[2]-bbox[0])/2, bbox[1]+(bbox[3]-bbox[1])/2]
                
                x1 = int(center[0]-length/2)
                x2 = int(center[0]+length/2)
                y1 = int(center[1]-length/2)
                y2 = int(center[1]+length/2)

                if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
                    continue
                
                cropped_faces.append(img[y1:y2, x1:x2])
                offset.append((x1, y1, length))

        if len(cropped_faces) != 0:
            faces = self.pre_process_batch(cropped_faces)
            faces = faces.to(DEVICE)

            _, _, poses = self.model(faces, faces)
            # pos_ploted = demoKpt(pos, img, is_render=False)
            end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1

            for idx in range(len(poses)):
                pos = poses[idx]
                pos = pos.cpu().squeeze().numpy().transpose(1,2,0)*280

                # pos = pos.cpu().squeeze().numpy().transpose(1,2,0)
                landmarks = getLandmark(pos)
                kpt = np.round(landmarks).astype(np.int32)
                for i in range(kpt.shape[0]):
                    st = kpt[i, :2].copy()
                    st[0] = st[0] / 256 * offset[idx][2] + offset[idx][0]
                    st[1] = st[1] / 256 * offset[idx][2] + offset[idx][1]
                    img = cv2.circle(img, (st[0], st[1]), 1, (0, 255, 0), 2)
                    if i in end_list:
                        continue

                    ed = kpt[i + 1, :2].copy() 
                    ed[0] = ed[0] / 256 * offset[idx][2] + offset[idx][0]
                    ed[1] = ed[1] / 256 * offset[idx][2] + offset[idx][1]
                    img = cv2.line(img, (st[0], st[1]), (ed[0], ed[1]), (0, 255, 0), 1)

        return img

    def pre_process_single_img(self, img: np.ndarray):
        # Resize
        try:
            img = cv2.resize(img, (256,256))
        except Exception as e:
            logging.error(e)
            import ipdb; ipdb.set_trace(context=10)

        # Normalize
        img = (img/255).astype(np.float32)
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / np.sqrt(img[:, :, i].var() + 0.001)
        
        # To tensor
        img = toTensor(img)

        return img

    def pre_process_batch(self, imgs: list):
        _imgs = [self.pre_process_single_img(img) for img in imgs]

        # import ipdb; ipdb.set_trace(context=10)
        return torch.stack(_imgs)