'''
functions about rendering mesh(from 3d obj to 2d image).
only use rasterization render here.
Note that:
1. Generally, render func includes camera, light, raterize. Here no camera and light(I write these in other files)
2. Generally, the input vertices are normalized to [-1,1] and cetered on [0, 0]. (in world space)
   Here, the vertices are using image coords, which centers on [weight/2, height/2] with the y-axis pointing to oppisite direction.
 Means: render here only conducts interpolation.(I just want to make the input flexible)

Author: Yao Feng 
Mail: yaofeng1995@gmail.com
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time

from .cython import mesh_core_cython

def rasterize_triangles(vertices, triangles, height, weight):
    ''' 
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
        height: height
        weight: width
    Returns:
        depth_buffer: [height, weight] saves the depth, here, the bigger the z, the fronter the point.
        triangle_buffer: [height, weight] saves the tri id(-1 for no triangle). 
        barycentric_weight: [height, weight, 3] saves corresponding barycentric weight.

    # Each triangle has 3 vertices & Each vertex has 3 coordinates x, y, z.
    # height, weight is the size of rendering
    '''

    # initial 
    depth_buffer = np.zeros([height, weight], dtype = np.float32) - 999999. #set the initial z to the farest position
    triangle_buffer = np.zeros([height, weight], dtype = np.int32) - 1  # if tri id = -1, the pixel has no triangle correspondance
    barycentric_weight = np.zeros([height, weight, 3], dtype = np.float32)  # 
    
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()

    mesh_core_cython.rasterize_triangles_core(
                vertices, triangles,
                depth_buffer, triangle_buffer, barycentric_weight, 
                vertices.shape[0], triangles.shape[0], 
                height, weight)

def render_colors(vertices, triangles, colors, height, weight, channel = 3, BG = None):
    ''' render mesh with colors
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3] 
        colors: [nver, 3]
        height: height
        weight: width  
        channel: channel
        BG: background image
    Returns:
        image: [height, weight, channel]. rendered image./rendering.
    '''

    # initial 
    if BG is None:
        image = np.zeros((height, weight, channel), dtype = np.float32)
    else:
        assert BG.shape[0] == height and BG.shape[1] == weight and BG.shape[2] == channel
        image = BG
    depth_buffer = np.zeros([height, weight], dtype = np.float32, order = 'C') - 999999.

    # change orders. --> C-contiguous order(column major)
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    colors = colors.astype(np.float32).copy()
    ###
    st = time()

    mesh_core_cython.render_colors_core(
                image, vertices, triangles,
                colors,
                depth_buffer,
                vertices.shape[0], triangles.shape[0], 
                height, weight, channel)
    return image


def render_texture(vertices, triangles, texture, tex_coords, tex_triangles, height, weight, channel = 3, mapping_type = 'nearest', BG = None):
    ''' render mesh with texture map
    Args:
        vertices: [3, nver]
        triangles: [3, ntri]
        texture: [tex_h, tex_w, 3]
        tex_coords: [ntexcoords, 3]
        tex_triangles: [ntri, 3]
        height: height of rendering
        weight: width of rendering
        channel: channel
        mapping_type: 'bilinear' or 'nearest'
    '''
    # initial 
    if BG is None:
        image = np.zeros((height, weight, channel), dtype = np.float32)
    else:
        assert BG.shape[0] == height and BG.shape[1] == weight and BG.shape[2] == channel
        image = BG

    depth_buffer = np.zeros([height, weight], dtype = np.float32, order = 'C') - 999999.
    
    tex_h, tex_w, tex_c = texture.shape
    if mapping_type == 'nearest':
        mt = int(0)
    elif mapping_type == 'bilinear':
        mt = int(1)
    else:
        mt = int(0)
    
    # -> C order
    vertices = vertices.astype(np.float32).copy()
    triangles = triangles.astype(np.int32).copy()
    texture = texture.astype(np.float32).copy()
    tex_coords = tex_coords.astype(np.float32).copy()
    tex_triangles = tex_triangles.astype(np.int32).copy()

    mesh_core_cython.render_texture_core(
                image, vertices, triangles,
                texture, tex_coords, tex_triangles,
                depth_buffer,
                vertices.shape[0], tex_coords.shape[0], triangles.shape[0], 
                height, weight, channel,
                tex_h, tex_w, tex_c,
                mt)
    return image

