import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform

from .data import FACE_MASK_MEAN_FIX_RATE, FACE_MASK_NP, UV_KPT, toTensor

weight_mask_np = io.imread('data/uv-data/uv_weight_mask.png').astype(float)
weight_mask_np[weight_mask_np == 255] = 256
weight_mask_np = weight_mask_np / 16

weight_mask = torch.from_numpy(weight_mask_np)
face_mask = torch.from_numpy(FACE_MASK_NP)
face_mask_3D = toTensor(np.repeat(np.reshape(FACE_MASK_NP, (256, 256, 1)), 3, -1))
foreface_ind = np.array(np.where(FACE_MASK_NP > 0)).T
if torch.cuda.is_available():
    weight_mask = weight_mask.cuda().float()
    face_mask = face_mask.cuda().float()
    face_mask_3D = face_mask_3D.cuda().float()


def UVLoss(is_foreface=False, is_weighted=False, is_nme=False):
    class TemplateLoss(nn.Module):
        def __init__(self, rate=1.0):
            super(TemplateLoss, self).__init__()
            self.rate = rate
            self.weight_mask = nn.Parameter(weight_mask.clone())
            self.face_mask = nn.Parameter(face_mask.clone())
            self.weight_mask.requires_grad = False
            self.face_mask.requires_grad = False

        def forward(self, y_true, y_pred):
            if is_nme:
                pred = y_pred[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
                gt = y_true[:, :, foreface_ind[:, 0], foreface_ind[:, 1]]
                for i in range(y_true.shape[0]):
                    pred[i, 2] = pred[i, 2] - torch.mean(pred[i, 2])
                    gt[i, 2] = gt[i, 2] - torch.mean(gt[i, 2])
                dist = torch.mean(torch.norm(pred - gt, dim=1), dim=1)
                left = torch.min(gt[:, 0, :], dim=1)[0]
                right = torch.max(gt[:, 0, :], dim=1)[0]
                top = torch.min(gt[:, 1, :], dim=1)[0]
                bottom = torch.max(gt[:, 1, :], dim=1)[0]
                bbox_size = torch.sqrt((right - left) * (bottom - top))
                dist = dist / bbox_size
                return torch.mean(dist) * self.rate

            dist = torch.sqrt(torch.sum((y_true - y_pred) ** 2, 1))
            if is_weighted:
                dist = dist * self.weight_mask
            if is_foreface:
                dist = dist * (self.face_mask * FACE_MASK_MEAN_FIX_RATE)

            loss = torch.mean(dist)
            return loss * self.rate

    return TemplateLoss


def getLossFunction(loss_func_name='SquareError'):
    if loss_func_name == 'RootSquareError' or loss_func_name == 'rse':
        return UVLoss(is_foreface=False, is_weighted=False)
    elif loss_func_name == 'WeightedRootSquareError' or loss_func_name == 'wrse':
        return UVLoss(is_foreface=False, is_weighted=True)
    elif loss_func_name == 'ForefaceRootSquareError' or loss_func_name == 'frse':
        return UVLoss(is_foreface=True, is_weighted=False)
    elif loss_func_name == 'ForefaceWeightedRootSquareError' or loss_func_name == 'fwrse':
        return UVLoss(is_foreface=True, is_weighted=True)
    elif loss_func_name == 'nme':
        return UVLoss(is_foreface=True, is_weighted=False, is_nme=True)
    else:
        print('unknown loss:', loss_func_name)


def PRNError(is_2d=False, is_normalized=True, is_foreface=True, is_landmark=False, is_gt_landmark=False):
    def templateError(y_gt, y_fit, bbox=None, landmarks=None):
        assert (not (is_foreface and is_landmark))
        y_true = y_gt.copy()
        y_pred = y_fit.copy()
        y_true[:, :, 2] = y_true[:, :, 2] * FACE_MASK_NP
        y_pred[:, :, 2] = y_pred[:, :, 2] * FACE_MASK_NP
        y_true_mean = np.mean(y_true[:, :, 2]) * FACE_MASK_MEAN_FIX_RATE
        y_pred_mean = np.mean(y_pred[:, :, 2]) * FACE_MASK_MEAN_FIX_RATE
        y_true[:, :, 2] = y_true[:, :, 2] - y_true_mean
        y_pred[:, :, 2] = y_pred[:, :, 2] - y_pred_mean

        if is_landmark:
            # the gt landmark is not the same as the landmarks get from mesh using index
            if is_gt_landmark:
                gt = landmarks.copy()
                gt[:, 2] = gt[:, 2] - gt[:, 2].mean()

                pred = y_pred[UV_KPT[:, 0], UV_KPT[:, 1]]
                diff = np.square(gt - pred)
                if is_2d:
                    dist = np.sqrt(np.sum(diff[:, 0:2], axis=-1))
                else:
                    dist = np.sqrt(np.sum(diff, axis=-1))
            else:
                gt = y_true[UV_KPT[:, 0], UV_KPT[:, 1]]
                pred = y_pred[UV_KPT[:, 0], UV_KPT[:, 1]]
                gt[:, 2] = gt[:, 2] - gt[:, 2].mean()
                pred[:, 2] = pred[:, 2] - pred[:, 2].mean()
                diff = np.square(gt - pred)
                if is_2d:
                    dist = np.sqrt(np.sum(diff[:, 0:2], axis=-1))
                else:
                    dist = np.sqrt(np.sum(diff, axis=-1))
        else:
            diff = np.square(y_true - y_pred)
            if is_2d:
                dist = np.sqrt(np.sum(diff[:, :, 0:2], axis=-1))
            else:
                # 3d
                dist = np.sqrt(np.sum(diff, axis=-1))
            if is_foreface:
                dist = dist * FACE_MASK_NP * FACE_MASK_MEAN_FIX_RATE

        if is_normalized:  # 2D bbox size
            # bbox_size = np.sqrt(np.sum(np.square(bbox[0, :] - bbox[1, :])))
            if is_landmark:
                bbox_size = np.sqrt((bbox[0, 0] - bbox[1, 0]) * (bbox[0, 1] - bbox[1, 1]))
            else:
                face_vertices = y_gt[FACE_MASK_NP > 0]
                minx, maxx = np.min(face_vertices[:, 0]), np.max(face_vertices[:, 0])
                miny, maxy = np.min(face_vertices[:, 1]), np.max(face_vertices[:, 1])
                llength = np.sqrt((maxx - minx) * (maxy - miny))
                bbox_size = llength
        else:  # 3D bbox size
            face_vertices = y_gt[FACE_MASK_NP > 0]
            minx, maxx = np.min(face_vertices[:, 0]), np.max(face_vertices[:, 0])
            miny, maxy = np.min(face_vertices[:, 1]), np.max(face_vertices[:, 1])
            minz, maxz = np.min(face_vertices[:, 2]), np.max(face_vertices[:, 2])
            if is_landmark:
                llength = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
            else:
                llength = np.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2 + (maxz - minz) ** 2)
            bbox_size = llength

        loss = np.mean(dist / bbox_size)
        return loss

    return templateError


def getErrorFunction(error_func_name='NME'):
    if error_func_name == 'nme2d' or error_func_name == 'normalized mean error2d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=True)
    elif error_func_name == 'nme3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=True)
    elif error_func_name == 'landmark2d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True)
    elif error_func_name == 'landmark3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True)
    elif error_func_name == 'gtlandmark2d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=True, is_normalized=True, is_foreface=False, is_landmark=True,
                        is_gt_landmark=True)
    elif error_func_name == 'gtlandmark3d' or error_func_name == 'normalized mean error3d':
        return PRNError(is_2d=False, is_normalized=True, is_foreface=False, is_landmark=True,
                        is_gt_landmark=True)
    else:
        print('unknown error:', error_func_name)
