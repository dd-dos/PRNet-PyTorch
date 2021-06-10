import math
import os
from math import acos, asin, atan, atan2, cos, sin

import cv2
import numpy as np
import skimage
import torch
from PIL import Image
from skimage import io

from . import faceutil
from .faceutil import mesh
from .faceutil.morphable_model import MorphabelModel
from .matlabutil import NormDirection

#  global data
BFM = MorphabelModel('data/Out/BFM.mat')
DEFAULT_INIT_IMAGE_SHAPE = np.array([450, 450, 3])
DEFAULT_CROPPED_IMAGE_SHAPE = np.array([256, 256, 3])
DEFAULT_UVMAP_SHAPE = np.array([256, 256, 3])
FACE_MASK_NP = io.imread('data/uv-data/uv_face_mask.png') / 255.
FACE_MASK_MEAN_FIX_RATE = (256 * 256) / np.sum(FACE_MASK_NP)


def custom_crop(img, bbox, ratio=1/4):
    if not isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().detach().numpy()

    bb_width, bb_height = bbox[2]-bbox[0], bbox[3]-bbox[1]
    img_width, img_height = img.size

    x1 = int(np.max([0, bbox[0]-bb_width*ratio]))
    x2 = int(np.min([img_width, bbox[2]+bb_width*ratio]))
    y1 = int(np.max([0, bbox[1]-bb_height*ratio]))
    y2 = int(np.min([img_height, bbox[3]+bb_height*ratio]))

    return img.crop((x1,y1,x2,y2))


def process_uv(uv_coordinates):
    [uv_h, uv_w, uv_c] = DEFAULT_UVMAP_SHAPE
    uv_coordinates[:, 0] = uv_coordinates[:, 0] * (uv_w - 1)
    uv_coordinates[:, 1] = uv_coordinates[:, 1] * (uv_h - 1)
    uv_coordinates[:, 1] = uv_h - uv_coordinates[:, 1] - 1
    uv_coordinates = np.hstack((uv_coordinates, np.zeros((uv_coordinates.shape[0], 1))))  # add z
    return uv_coordinates


def toTensor(image):
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = torch.from_numpy(image)
    return image


def readUVKpt(uv_kpt_path):
    file = open(uv_kpt_path, 'r', encoding='utf-8')
    lines = file.readlines()
    # txt is inversed
    x_line = lines[1]
    y_line = lines[0]
    UV_KPT = np.zeros((68, 2)).astype(int)
    x_tokens = x_line.strip().split(' ')
    y_tokens = y_line.strip().split(' ')
    for i in range(68):
        UV_KPT[i][0] = int(float(x_tokens[i]))
        UV_KPT[i][1] = int(float(y_tokens[i]))
    return UV_KPT

# global data
UV_COORDS_TEMP = faceutil.morphable_model.load.load_uv_coords('data/Out/BFM_UV.mat')
UV_COORDS = process_uv(UV_COORDS_TEMP)
UV_KPT = readUVKpt('data/uv-data/uv_kpt_ind.txt')
UVMAP_PLACE_HOLDER = np.ones((256, 256, 1))


def getLandmark(ipt):
    # from uv map
    kpt = ipt[UV_KPT[:, 0], UV_KPT[:, 1]]
    return kpt


def bfm2Mesh(bfm_info, image_shape=DEFAULT_INIT_IMAGE_SHAPE):
    """
    generate mesh data from 3DMM (bfm2009) parameters
    :param bfm_info:
    :param image_shape:
    :return: meshe data
    """
    [image_h, image_w, channel] = image_shape
    pose_para = bfm_info['Pose_Para'].T.astype(np.float32)
    shape_para = bfm_info['Shape_Para'].astype(np.float32)
    exp_para = bfm_info['Exp_Para'].astype(np.float32)
    tex_para = bfm_info['Tex_Para'].astype(np.float32)
    color_Para = bfm_info['Color_Para'].astype(np.float32)
    illum_Para = bfm_info['Illum_Para'].astype(np.float32)

    # 2. generate mesh_numpy
    # shape & exp param
    vertices = BFM.generate_vertices(shape_para, exp_para)
    # texture param
    tex = BFM.generate_colors(tex_para)
    norm = NormDirection(vertices, BFM.model['tri'])

    # color param
    [Gain_r, Gain_g, Gain_b, Offset_r, Offset_g, Offset_b, c] = color_Para[0]
    M = np.array([[0.3, 0.59, 0.11], [0.3, 0.59, 0.11], [0.3, 0.59, .11]])

    g = np.diag([Gain_r, Gain_g, Gain_b])
    o = [Offset_r, Offset_g, Offset_b]
    o = np.tile(o, (vertices.shape[0], 1))

    # illum param
    [Amb_r, Amb_g, Amb_b, Dir_r, Dir_g, Dir_b, thetal, phil, ks, v] = illum_Para[0]
    Amb = np.diag([Amb_r, Amb_g, Amb_b])
    Dir = np.diag([Dir_r, Dir_g, Dir_b])
    l = np.array([math.cos(thetal) * math.sin(phil), math.sin(thetal), math.cos(thetal) * math.cos(phil)]).T
    h = l + np.array([0, 0, 1]).T
    h = h / math.sqrt(h.T.dot(h))

    # final color
    n_l = l.T.dot(norm.T)
    n_h = h.T.dot(norm.T)
    n_l = np.array([max(x, 0) for x in n_l])
    n_h = np.array([max(x, 0) for x in n_h])
    n_l = np.tile(n_l, (3, 1))
    n_h = np.tile(n_h, (3, 1))
    L = Amb.dot(tex.T) + Dir.dot(n_l * tex.T) + (ks * Dir).dot((n_h ** v))
    CT = g.dot(c * np.eye(3) + (1 - c) * M)
    tex_color = CT.dot(L) + o.T
    tex_color = np.minimum(np.maximum(tex_color, 0), 1).T

    # transform mesh_numpy
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]

    # 3ddfa-R: radian || normal transform - R:degree
    transformed_vertices = BFM.transform_3ddfa(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
    image_vertices = projected_vertices.copy()
    # should not -1
    image_vertices[:, 1] = image_h - image_vertices[:, 1]
    mesh_info = {'vertices': image_vertices, 'triangles': BFM.full_triangles,
                 'full_triangles': BFM.full_triangles,
                 'colors': tex_color}
    # 'landmarks': bfm_info['pt3d_68'].T
    return mesh_info


def UVmap2Mesh(uv_position_map, uv_texture_map=None, only_foreface=True, is_extra_triangle=False):
    """
    if no texture map is provided, translate the position map to a point cloud
    :param uv_position_map:
    :param uv_texture_map:
    :param only_foreface:
    :return: mesh data
    """
    [uv_h, uv_w, uv_c] = DEFAULT_UVMAP_SHAPE
    vertices = []
    colors = []
    triangles = []
    if uv_texture_map is not None:
        for i in range(uv_h):
            for j in range(uv_w):
                if not only_foreface:
                    vertices.append(uv_position_map[i][j])
                    colors.append(uv_texture_map[i][j])
                    pa = i * uv_h + j
                    pb = i * uv_h + j + 1
                    pc = (i - 1) * uv_h + j
                    pd = (i + 1) * uv_h + j + 1
                    if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                        triangles.append([pa, pb, pc])
                        triangles.append([pa, pc, pb])
                        triangles.append([pa, pb, pd])
                        triangles.append([pa, pd, pb])

                else:
                    if FACE_MASK_NP[i, j] == 0:
                        vertices.append(np.array([0, 0, 0]))
                        colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        vertices.append(uv_position_map[i][j])
                        colors.append(uv_texture_map[i][j])
                        pa = i * uv_h + j
                        pb = i * uv_h + j + 1
                        pc = (i - 1) * uv_h + j
                        pd = (i + 1) * uv_h + j + 1
                        if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                            if is_extra_triangle:
                                pe = (i - 1) * uv_h + j + 1
                                pf = (i + 1) * uv_h + j
                                if (FACE_MASK_NP[i, j + 1] > 0) and (FACE_MASK_NP[i + 1, j + 1] > 0) and (FACE_MASK_NP[i + 1, j] > 0) and (
                                        FACE_MASK_NP[i - 1, j + 1] > 0 and FACE_MASK_NP[i - 1, j] > 0):
                                    triangles.append([pa, pb, pc])
                                    triangles.append([pa, pc, pb])
                                    triangles.append([pa, pc, pe])
                                    triangles.append([pa, pe, pc])
                                    triangles.append([pa, pb, pe])
                                    triangles.append([pa, pe, pb])
                                    triangles.append([pb, pc, pe])
                                    triangles.append([pb, pe, pc])

                                    triangles.append([pa, pb, pd])
                                    triangles.append([pa, pd, pb])
                                    triangles.append([pa, pb, pf])
                                    triangles.append([pa, pf, pb])
                                    triangles.append([pa, pd, pf])
                                    triangles.append([pa, pf, pd])
                                    triangles.append([pb, pd, pf])
                                    triangles.append([pb, pf, pd])

                            else:
                                if not FACE_MASK_NP[i, j + 1] == 0:
                                    if not FACE_MASK_NP[i - 1, j] == 0:
                                        triangles.append([pa, pb, pc])
                                        triangles.append([pa, pc, pb])
                                    if not FACE_MASK_NP[i + 1, j + 1] == 0:
                                        triangles.append([pa, pb, pd])
                                        triangles.append([pa, pd, pb])
    else:
        for i in range(uv_h):
            for j in range(uv_w):
                if not only_foreface:
                    vertices.append(uv_position_map[i][j])
                    colors.append(np.array([64, 64, 64]))
                    pa = i * uv_h + j
                    pb = i * uv_h + j + 1
                    pc = (i - 1) * uv_h + j
                    if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                        triangles.append([pa, pb, pc])
                else:
                    if FACE_MASK_NP[i, j] == 0:
                        vertices.append(np.array([0, 0, 0]))
                        colors.append(np.array([0, 0, 0]))
                        continue
                    else:
                        vertices.append(uv_position_map[i][j])
                        colors.append(np.array([128, 0, 128]))
                        pa = i * uv_h + j
                        pb = i * uv_h + j + 1
                        pc = (i - 1) * uv_h + j
                        if (i > 0) & (i < uv_h - 1) & (j < uv_w - 1):
                            if not FACE_MASK_NP[i, j + 1] == 0:
                                if not FACE_MASK_NP[i - 1, j] == 0:
                                    triangles.append([pa, pb, pc])
                                    triangles.append([pa, pc, pb])

    vertices = np.array(vertices)
    colors = np.array(colors)
    triangles = np.array(triangles)
    # verify_face = mesh.render.render_colors(verify_vertices, verify_triangles, verify_colors, height, width,
    #                                         channel)
    mesh_info = {'vertices': vertices, 'triangles': triangles,
                 'full_triangles': triangles,
                 'colors': colors}
    return mesh_info


def mesh2UVmap(mesh_data):
    """
    generate uv map from mesh data
    :param mesh_data:
    :return: uv position map and corresponding texture
    """
    [uv_h, uv_w, uv_c] = DEFAULT_UVMAP_SHAPE
    vertices = mesh_data['vertices']
    colors = mesh_data['colors']
    triangles = mesh_data['full_triangles']
    # colors = colors / np.max(colors)
    # model_image = mesh.render.render_colors(vertices, BFM.triangles, colors, image_h, image_w) # only for show

    uv_texture_map = mesh.render.render_colors(UV_COORDS, triangles, colors, uv_h, uv_w, uv_c)
    position = vertices.copy()
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z
    uv_position_map = mesh.render.render_colors(UV_COORDS, triangles, position, uv_h, uv_w, uv_c)
    return uv_position_map, uv_texture_map


def renderMesh(mesh_info, image_shape=None):
    if image_shape is None:
        image_height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
        image_width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
    else:
        [image_height, image_width, image_channel] = image_shape
    mesh_image = mesh.render.render_colors(mesh_info['vertices'],
                                           mesh_info['triangles'],
                                           mesh_info['colors'], image_height, image_width)
    mesh_image = np.clip(mesh_image, 0., 1.)
    return mesh_image


def getTransformMatrix(s, angles, t, height):
    x, y, z = angles[0], angles[1], angles[2]

    Rx = np.array([[1, 0, 0],
                   [0, cos(x), sin(x)],
                   [0, -sin(x), cos(x)]])
    Ry = np.array([[cos(y), 0, -sin(y)],
                   [0, 1, 0],
                   [sin(y), 0, cos(y)]])
    Rz = np.array([[cos(z), sin(z), 0],
                   [-sin(z), cos(z), 0],
                   [0, 0, 1]])
    # rotate
    R = Rx.dot(Ry).dot(Rz)
    R = R.astype(np.float32)
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[3, 3] = 1.
    # scale
    S = np.diagflat([s, s, s, 1.])
    T = S.dot(T)
    # offset move
    M = np.diagflat([1., 1., 1., 1.])
    M[0:3, 3] = t.astype(np.float32)
    T = M.dot(T)
    # revert height
    # x[:,1]=height-x[:,1]
    H = np.diagflat([1., 1., 1., 1.])
    H[1, 1] = -1.0
    H[1, 3] = height
    T = H.dot(T)
    return T.astype(np.float32)


def getColors(image, posmap):
    [h, w, _] = image.shape
    [uv_h, uv_w, uv_c] = posmap.shape
    # tex = np.zeros((uv_h, uv_w, uv_c))
    around_posmap = np.around(posmap).clip(0, h - 1).astype(np.int)

    tex = image[around_posmap[:, :, 1], around_posmap[:, :, 0], :]
    return tex
