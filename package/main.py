import logging

import cv2
import numpy as np
import torch
from mlchain.decorators import except_serving
from models.resfcn256 import ResFCN256
from utils.data import getLandmark, toTensor

logging.getLogger().setLevel(logging.INFO)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class FacePatternModel: 
    def __init__(self, model_path: str):
        '''
        Main class for extracting face pattern.

        :model_path: path to model weights. Currently support PRnet.
        '''
        self.model = ResFCN256()
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

    def draw_landmarks(self, img: np.ndarray, bboxes: list):
        end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1

        kpts, offset = self.get_3D_landmarks_bboxes(img, bboxes)
        img = self.draw_bboxes(img, bboxes)

        for idx in range(len(offset)):
            kpt = np.round(kpts[f'{idx}']).astype(np.uint32)
            for i in range(len(kpt)):
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

    @except_serving
    def pre_process_single_img(self, img: np.ndarray):
        # Resize
        img = cv2.resize(img, (256,256))

        # Normalize
        img = (img/255).astype(np.float32)
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / np.sqrt(img[:, :, i].var() + 0.001)
        
        # To tensor
        img = toTensor(img)

        return img

    @except_serving
    def pre_process_batch(self, imgs: list):
        _imgs = [self.pre_process_single_img(img) for img in imgs]

        # import ipdb; ipdb.set_trace(context=10)
        return torch.stack(_imgs)

    @torch.no_grad()
    def get_3D_landmarks_bboxes(self, img: np.ndarray, bboxes: list):
        '''
        Get 3D landmarks of a single image with 1 or multiple bounding boxes.
        '''
        cropped_faces = []
        height, width, _ = img.shape
        offset = []
        landmarks_list = {}

        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
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

            for idx in range(len(poses)):
                pos = poses[idx]
                pos = pos.cpu().squeeze().numpy().transpose(1,2,0)*280

                # pos = pos.cpu().squeeze().numpy().transpose(1,2,0)
                landmarks_list[f'{idx}'] = getLandmark(pos)
                
        return landmarks_list, offset
    
    @torch.no_grad()
    def get_3D_landmarks_face(self, face: np.ndarray):
        face = self.pre_process_single_img(face)
        face = face.to(DEVICE).unsqueeze(0)
        _, _, poses = self.model(face, face)

        pos = poses.cpu().squeeze().numpy().transpose(1,2,0)*280

        lms = getLandmark(pos)

        return lms

    def draw_bboxes(self, img: np.ndarray, bboxes: list):
        height, width, _ = img.shape

        for idx in range(len(bboxes)):
            bbox = bboxes[idx]
            if bbox[-1] >= 0.9:
                length = max(bbox[2]-bbox[0], bbox[3]-bbox[1])*1.2
                center = [bbox[0]+(bbox[2]-bbox[0])/2, bbox[1]+(bbox[3]-bbox[1])/2]
                
                x1 = int(center[0]-length/2)
                x2 = int(center[0]+length/2)
                y1 = int(center[1]-length/2)
                y2 = int(center[1]+length/2)

                if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
                    continue

                img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255,0,0), thickness=1)

        return img                

