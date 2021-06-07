# from data import BFM, modelParam2Mesh, UVMap2Mesh
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrender
import scipy.io as sio
import torch
import torchvision
import tqdm
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import io, transform

from utils.augmentation import rotateData

from .data import BFM, UV_KPT, UVmap2Mesh, bfm2Mesh, getLandmark, mesh2UVmap
from .faceutil import mesh

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1

IMAGE_WIDTH = 256
r = pyrender.OffscreenRenderer(IMAGE_WIDTH, IMAGE_WIDTH)
scene = pyrender.Scene()


def light_test(vertices, light_positions, light_intensities, triangles, colors, bg=None, h=256, w=256):
    lit_colors = mesh.light.add_light(vertices, triangles, colors, light_positions, light_intensities)
    # image_vertices = mesh.transform.to_image(vertices, h, w)
    rendering = mesh.render.render_colors(vertices, triangles, lit_colors, h, w, BG=bg)
    # rendering = np.minimum((np.maximum(rendering, 0)), 1)
    return rendering


def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:

        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], 255 - vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)


def renderLightBack(posmap, init_image=None):
    tex = np.ones((256, 256, 3)) / 2
    mesh = UVmap2Mesh(posmap, tex)
    vertices = mesh['vertices']
    triangles = mesh['triangles']
    colors = mesh['colors'] / np.max(mesh['colors'])
    showMesh(mesh)

    light_intensities = np.array([[1, 1, 1]])
    for i, p in enumerate(range(-200, 201, 60)):
        light_positions = np.array([[p, -100, 300]])
        image = light_test(vertices, light_positions, light_intensities, triangles, colors, bg=init_image)
        showImage(image)


def renderLight(posmap, init_image=None, is_render=True):
    tex = np.ones((256, 256, 3)) / 2
    mesh = UVmap2Mesh(posmap, tex, is_extra_triangle=False)
    vertices = mesh['vertices']
    triangles = mesh['triangles']
    colors = mesh['colors'] / np.max(mesh['colors'])
    file = 'tmp/light/test.obj'
    write_obj_with_colors(file, vertices, triangles, colors)

    obj = trimesh.load(file)
    # obj.visual.vertex_colors = np.random.uniform(size=obj.vertices.shape)
    obj.visual.face_colors = np.array([0.05, 0.1, 0.2])

    mesh = pyrender.Mesh.from_trimesh(obj, smooth=False)

    scene.add(mesh, pose=np.eye(4))

    camera_pose = np.eye(4)
    camera_pose[0, 3] = 128
    camera_pose[1, 3] = 128
    camera_pose[2, 3] = 300
    camera = pyrender.OrthographicCamera(xmag=128, ymag=128, zfar=1000)

    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=8.0)
    scene.add(light, pose=camera_pose)
    color, depth = r.render(scene)
    if is_render:
        plt.imshow(color)
        plt.show()

    if init_image is not None:
        sum_mask = np.mean(color, axis=-1)
        fuse_img = color.copy()
        fuse_img[sum_mask > 128] = init_image[sum_mask > 128]
        if is_render:
            plt.imshow(fuse_img)
            plt.show()
        scene.clear()
        return fuse_img

    scene.clear()
    return color


def plot_kpt(image, kpt, is_render=True, color_rate=0):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (0 + color_rate, 0, 255 - color_rate), 2)
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (0 + color_rate, 0, 255 - color_rate), 1)
    if is_render:
        showImage(image)
    return image


def demoKpt(posmap, image, is_render=True):
    kpt = getLandmark(posmap)
    ploted = plot_kpt(image, kpt, is_render=is_render)
    return ploted


def compareKpt(posmap, gtposmap, image, is_render=True):
    kpt1 = getLandmark(posmap)
    kpt2 = getLandmark(gtposmap)
    ploted = plot_kpt(image, kpt1, is_render=is_render)
    ploted = plot_kpt(ploted, kpt2, is_render=is_render, color_rate=int(255))
    return ploted


def demoAll(posmap, image, is_render=True):
    return renderLight(posmap, image.copy(), is_render=is_render), demoKpt(posmap, image.copy(), is_render=is_render)


def concatenate_compareKpt(pos, gtpos, image):
    pos_ploted = demoKpt(pos, image, is_render=False)
    gtpos_ploted = demoKpt(gtpos, image, is_render=False)

    concatenate_ploted = cv2.hconcat([gtpos_ploted, pos_ploted])

    return concatenate_ploted


def logTrainingSamples(gtposes, poses, metas, epoch, writer):
    for idx in range(poses.shape[0]):
        gtpos = gtposes[idx].squeeze().numpy().transpose(1,2,0)*280
        pos = poses[idx].squeeze().numpy().transpose(1,2,0)*280

        img_path = metas['img_path'][idx]
        img = cv2.imread(img_path)
        rotate_angle = metas['rotate_angle'][idx]

        dummy = np.zeros((256,256,3))
        img, _, _ = rotateData(img, dummy, specify_angle=rotate_angle)

        comparision = concatenate_compareKpt(pos, gtpos, img)
        comparision = cv2.cvtColor(comparision, cv2.COLOR_BGR2RGB)
        comparision = torch.tensor(comparision.transpose(2,0,1) / 255.0, 
                                   dtype=torch.float32).unsqueeze(0)

        grid = torchvision.utils.make_grid(comparision)
        writer.add_image(f'comparision-{idx}', grid, epoch)


def showTriangularMesh(vertices: np.ndarray, triangles: np.ndarray):
    num_triangles = triangles.shape[0]
    _triangles = triangles.transpose(1,0)
    _vertices = []
    for i in tqdm.tqdm(range(num_triangles)):
        tri_p0_ind, tri_p1_ind, tri_p2_ind = triangles[i]
        _vertices += [vertices[tri_p0_ind],
                      vertices[tri_p1_ind],
                      vertices[tri_p2_ind]]
        
    _vertices = np.array(_vertices).transpose(1,0)

    ax = plt.gca(projection="3d")
    tri_idx = [(3*i, 3*i+1, 3*i+2) for i in range(num_triangles)]
    ax.plot_trisurf(_vertices[0], _vertices[1], _vertices[2], triangles=tri_idx)     
    
    plt.show()
    plt.close()
    plt.clf()


def showVertices(vertices: np.ndarray, v_type='3D'):
    if v_type=='3D':
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        _vertices = vertices.transpose(1, 0)
        ax.scatter(_vertices[0],
                   _vertices[1],
                   _vertices[2],
                   marker=".")

        plt.show() 
        plt.close()
    elif v_type=='2D':
        # ax, fig = plt.figure()

        # _vertices = vertices.transpose(1, 0)
        # ax.scatter(_vertices[0],
        #            _vertices[1],
        #            marker=".")
        _vertices = vertices.transpose(1, 0)
        plt.scatter(_vertices[0],
                    _vertices[1],
                    marker='.')
        plt.show() 
        plt.close()        
    else:
        return


def showLandmark(image, kpt):
    kpt = np.round(kpt).astype(np.int)
    image[kpt[:, 1], kpt[:, 0]] = np.array([1, 0, 0])
    image[kpt[:, 1] + 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt[:, 1] - 1, kpt[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt[:, 1] - 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
    image[kpt[:, 1] + 1, kpt[:, 0] - 1] = np.array([1, 0, 0])
    plt.imshow(image)
    plt.show()


def showLandmark2(image, kpt1, kpt2):
    kpt1 = np.round(kpt1).astype(np.int)
    kpt2 = np.round(kpt2).astype(np.int)
    image[kpt1[:, 1], kpt1[:, 0]] = np.array([1, 0, 0])
    image[kpt1[:, 1] + 1, kpt1[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt1[:, 1] - 1, kpt1[:, 0] + 1] = np.array([1, 0, 0])
    image[kpt1[:, 1] - 1, kpt1[:, 0] - 1] = np.array([1, 0, 0])
    image[kpt1[:, 1] + 1, kpt1[:, 0] - 1] = np.array([1, 0, 0])

    image[kpt2[:, 1], kpt2[:, 0]] = np.array([0, 1, 0])
    image[kpt2[:, 1] + 1, kpt2[:, 0]] = np.array([0, 1, 0])
    image[kpt2[:, 1] - 1, kpt2[:, 0]] = np.array([0, 1, 0])
    image[kpt2[:, 1], kpt2[:, 0] + 1] = np.array([0, 1, 0])
    image[kpt2[:, 1], kpt2[:, 0] - 1] = np.array([0, 1, 0])

    plt.imshow(image)
    plt.show()


def showGTLandmark(image_path):
    image = io.imread(image_path) / 255.0
    bfm_info = sio.loadmat(image_path.replace('jpg', 'mat'))
    if 'pt3d_68' in bfm_info.keys():
        kpt = bfm_info['pt3d_68'].T
    else:
        kpt = bfm_info['pt2d'].T
    showLandmark(image, kpt)

    mesh_info = bfm2Mesh(bfm_info, image.shape)

    kpt2 = mesh_info['vertices'][BFM.kpt_ind]
    showLandmark2(image, kpt, kpt2)
    return kpt, kpt2


def showImage(image, is_path=False):
    if is_path:
        img = io.imread(image) / 255.
        io.imshow(img)
        plt.show()
    else:
        io.imshow(image)
        plt.show()


def showMesh(mesh_info, init_img=None):
    height = np.ceil(np.max(mesh_info['vertices'][:, 1])).astype(int)
    width = np.ceil(np.max(mesh_info['vertices'][:, 0])).astype(int)
    channel = 3
    if init_img is not None:
        [height, width, channel] = init_img.shape
    mesh_image = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                           height, width, channel)
    if init_img is None:
        io.imshow(mesh_image)
        plt.show()
    else:
        plt.subplot(1, 3, 1)
        plt.imshow(mesh_image)

        plt.subplot(1, 3, 3)
        plt.imshow(init_img)

        verify_img = mesh.render.render_colors(mesh_info['vertices'], mesh_info['triangles'], mesh_info['colors'],
                                               height, width, channel, BG=init_img)
        plt.subplot(1, 3, 2)
        plt.imshow(verify_img)

        plt.show()


def show(ipt, is_file=False, mode='image'):
    if mode == 'image':
        if is_file:
            # ipt is a path
            image = io.imread(ipt) / 255.
        else:
            image = ipt
        io.imshow(image)
        plt.show()
    elif mode == 'uvmap':
        # ipt should be [posmap texmap] or [posmap texmap image]
        assert (len(ipt) > 1)
        init_image = None
        if is_file:
            uv_position_map = np.load(ipt[0])
            uv_texture_map = io.imread(ipt[1]) / 255.
            if len(ipt) > 2:
                init_image = io.imread(ipt[2]) / 255.
        else:
            uv_position_map = ipt[0]
            uv_texture_map = ipt[1]
            if len(ipt) > 2:
                init_image = ipt[2]
        mesh_info = UVmap2Mesh(uv_position_map=uv_position_map, uv_texture_map=uv_texture_map)
        showMesh(mesh_info, init_image)
    elif mode == 'mesh':
        if is_file:
            if len(ipt) == 2:
                mesh_info = sio.loadmat(ipt[0])
                init_image = io.imread(ipt[1]) / 255.
            else:
                mesh_info = sio.loadmat(ipt)
                init_image = None
        else:
            if len(ipt == 2):
                mesh_info = ipt[0]
                init_image = ipt[1]
            else:
                mesh_info = ipt
                init_image = None
        showMesh(mesh_info, init_image)


if __name__ == "__main__":
    pass
    # showUVMap('data/images/AFLW2000-out/image00002/image00002_uv_posmap.npy', None,
    #           # 'data/images/AFLW2000-output/image00002/image00002_uv_texture_map.jpg',
    #           'data/images/AFLW2000-out/image00002/image00002_init.jpg', True)
    # show(['data/images/AFLW2000-crop-offset/image00002/image00002_cropped_uv_posmap.npy',
    #       'data/images/AFLW2000-crop/image00002/image00002_uv_texture_map.jpg',
    #       'data/images/AFLW2000-crop-offset/image00002/image00002_cropped.jpg'], is_file=True, mode='uvmap')
