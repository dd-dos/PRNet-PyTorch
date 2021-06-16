import numpy as np
import math
from PIL import ImageEnhance, Image
import cv2
import random
import imgaug.augmenters as iaa
from numpy.lib.type_check import imag

seq = iaa.Sequential([
    iaa.Sometimes(0.1, iaa.MotionBlur(k=(5, 13), angle=(-45, 45))),

    # Low resolution
    iaa.Sometimes(0.15, 	
        iaa.OneOf([	
            iaa.imgcorruptlike.Pixelate(severity=(2, 4)),	
            iaa.imgcorruptlike.JpegCompression(severity=(1, 2)),	
            iaa.KMeansColorQuantization(n_colors=(230, 256)),	
            iaa.UniformColorQuantization(n_colors=(30, 256)),	
        ])	
    ),

    # Low light condition
    iaa.Sometimes(0.18, 	
        iaa.Sequential([	
            iaa.JpegCompression(compression=(50, 90)),	
            iaa.OneOf([	
                iaa.AdditivePoissonNoise((1, 10), per_channel=True),	
                iaa.AdditivePoissonNoise((1, 5)),	
                iaa.AdditiveLaplaceNoise(scale=(0.005*255, 0.02*255)),	
                iaa.AdditiveLaplaceNoise(scale=(0.005*255, 0.02*255), per_channel=True)	
            ])	
        ])	
    ),

    # Heavy blur
    iaa.Sometimes(0.1, iaa.MotionBlur(k=(5, 13), angle=(-45, 45))),
    
    iaa.Sometimes(0.05,
        iaa.OneOf([
            iaa.GaussianBlur((4.0, 11.0)),
            iaa.AverageBlur(k=(7, 13)),
            iaa.MedianBlur(k=(7, 13)),
        ]),
    ),

    # iaa.Sometimes(0.1, iaa.PerspectiveTransform(scale=(0.01, 0.15), keep_size=True)),
    iaa.Sometimes(0.1, iaa.ChangeColorTemperature((5000, 12000))),
    iaa.Sometimes(0.1, iaa.CoarseDropout(0.02, size_percent=0.01, per_channel=1)),

    iaa.Sometimes(0.1, 
        iaa.OneOf([
            iaa.LinearContrast((1.25, 2.5)),
            iaa.LogContrast(gain=(0.5, 2)),
            iaa.SigmoidContrast(gain=7, cutoff=(0.1, 0.9))#, per_channel=True)
        ])
    ),
    iaa.Sometimes(0.2, iaa.Multiply((0.75, 1.25))),
    
    iaa.Sometimes(0.5, iaa.ChannelShuffle(p=1)),
])

def randomColor(image):
    """
    """
    PIL_image = Image.fromarray((image*255).astype(np.uint8))
    random_factor = np.random.randint(0, 31) / 10.
    color_image = ImageEnhance.Color(PIL_image).enhance(random_factor)  # 调整图像的饱和度
    
    random_factor = np.random.randint(10, 21) / 10.
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    
    random_factor = np.random.randint(10, 21) / 10.
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    
    random_factor = np.random.randint(0, 31) / 10.
    out = np.array(ImageEnhance.Sharpness(contrast_image).enhance(random_factor))
    
    out = out/255.
    return out


def getRotateMatrix(angle, image_shape):
    [image_height, image_width, image_channel] = image_shape
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(angle), math.sin(angle), 0], [math.sin(-angle), math.cos(angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rt_mat = t2.dot(r1).dot(t1)
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(-angle), math.sin(-angle), 0], [math.sin(angle), math.cos(-angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rt_mat_inv = t2.dot(r1).dot(t1)
    return rt_mat.astype(np.float32), rt_mat_inv.astype(np.float32)


def getRotateMatrix3D(angle, image_shape):
    [image_height, image_width, image_channel] = image_shape
    t1 = np.array([[1, 0, 0, -image_height / 2.], [0, 1, 0, -image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    r1 = np.array([[math.cos(angle), math.sin(angle), 0, 0], [math.sin(-angle), math.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t2 = np.array([[1, 0, 0, image_height / 2.], [0, 1, 0, image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_mat = t2.dot(r1).dot(t1)
    t1 = np.array([[1, 0, 0, -image_height / 2.], [0, 1, 0, -image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    r1 = np.array([[math.cos(-angle), math.sin(-angle), 0, 0], [math.sin(angle), math.cos(-angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t2 = np.array([[1, 0, 0, image_height / 2.], [0, 1, 0, image_width / 2.], [0, 0, 1, 0], [0, 0, 0, 1]])
    rt_mat_inv = t2.dot(r1).dot(t1)
    return rt_mat.astype(np.float32), rt_mat_inv.astype(np.float32)


# @numba.jit(numba.float32(numba.float32,numba.float32))
def myDot(a, b):
    return np.dot(a, b)


def rotateData(x, y, angle_range=45, specify_angle=None):
    if specify_angle is None:
        angle = np.random.randint(-angle_range, angle_range)
        angle = angle / 180. * np.pi
    else:
        angle = specify_angle
    [image_height, image_width, image_channel] = x.shape
    # move-rotate-move
    [rform, rform_inv] = getRotateMatrix(angle, x.shape)

    # rotate_x = transform.warp(x, rform_inv,
    #                           output_shape=(image_height, image_width))
    rotate_x = cv2.warpPerspective(x, rform, (image_height, image_width))
    rotate_y = y.copy()
    rotate_y[:, :, 2] = 1.
    rotate_y = rotate_y.reshape(image_width * image_height, image_channel)
    # rotate_y = rotate_y.dot(rform.T)
    rotate_y = myDot(rotate_y, rform.T)
    rotate_y = rotate_y.reshape(image_height, image_width, image_channel)
    rotate_y[:, :, 2] = y[:, :, 2]
    # for i in range(image_height):
    #     for j in range(image_width):
    #         rotate_y[i][j][2] = 1.
    #         rotate_y[i][j] = rotate_y[i][j].dot(rform.T)
    #         rotate_y[i][j][2] = y[i][j][2]
    # tex = np.ones((256, 256, 3))
    # from visualize import show
    # show([rotate_y, tex, rotate_x.astype(np.float32)], mode='uvmap')
    return rotate_x, rotate_y, angle


def gaussNoise(x, mean=0, var=0.001):
    noise = np.random.normal(mean, var ** 0.5, x.shape)
    out = x + noise
    out = np.clip(out, 0., 1.0)
    # cv.imshow("gasuss", out)
    return out


def randomErase(x, max_num=4, s_l=0.02, s_h=0.3, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=1.0):
    [img_h, img_w, img_c] = x.shape
    out = x.copy()
    num = np.random.randint(1, max_num)

    for i in range(num):
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)
        mask = np.zeros((img_h, img_w))
        mask[top:min(top + h, img_h), left:min(left + w, img_w)] = 1
        if np.random.rand() < 0.25:
            c = np.random.uniform(v_l, v_h)
            out[mask > 0] = c
        elif np.random.rand() < 0.75:
            c0 = np.random.uniform(v_l, v_h)
            c1 = np.random.uniform(v_l, v_h)
            c2 = np.random.uniform(v_l, v_h)
            out0 = out[:, :, 0]
            out0[mask > 0] = c0
            out1 = out[:, :, 1]
            out1[mask > 0] = c1
            out2 = out[:, :, 2]
            out2[mask > 0] = c2
        else:
            c0 = np.random.uniform(v_l, v_h)
            c1 = np.random.uniform(v_l, v_h)
            c2 = np.random.uniform(v_l, v_h)
            out0 = out[:, :, 0]
            out0[mask > 0] *= c0
            out1 = out[:, :, 1]
            out1[mask > 0] *= c1
            out2 = out[:, :, 2]
            out2[mask > 0] *= c2
    return out


def channelScale(x, min_rate=0.6, max_rate=1.4):
    out = x.copy()
    for i in range(3):
        r = np.random.uniform(min_rate, max_rate)
        out[:, :, i] = out[:, :, i] * r
    out = np.clip(out, 0, 1)
    return out


def cropRange(image, ratio=1/4):
    out = image.copy()
    width, height, _ = image.shape

    max_width_crop = int(width*ratio)
    max_height_crop = int(height*ratio)

    x_max_index = width - max_width_crop 
    y_max_index = height - max_height_crop 

    x_crop_length = random.randint(int(max_width_crop/2), max_width_crop)
    y_crop_length = random.randint(int(max_height_crop/2), max_height_crop)

    def cropWidth(img):
        x_rd = np.random.rand()
        if x_rd > 0.5:
            img[0:x_crop_length, :, :] = 0.
        else:
            img[width-x_crop_length:, :, :] = 0.

        return img

    def cropHeight(img):
        y_rd = np.random.rand()
        if y_rd > 0.5:
            img[:, 0:y_crop_length, :] = 0.
        else:
            img[:, height-y_crop_length:, :] = 0.

        return img

    rd = np.random.rand()
    if 0.35 > rd >= 0:
        out = cropWidth(out)
    elif 0.7 > rd >= 0.35:
        out = cropHeight(out)
    else:
        out = cropWidth(out)
        out = cropHeight(out)

    return out


def prnAugment_torch(x, y):
    out = x.copy().astype(np.float64)
    
    rd = np.random.rand()
    if 0.3 > rd >= 0:
        out = randomErase(out)
    elif 0.6 > rd >= 0.3:
        out = cropRange(out, ratio=1/4)

    out = out.astype(np.uint8)
    out = seq(image=out)
    
    rotate_angle = 0
    if np.random.rand() > 0.5:
        out, y, rotate_angle = rotateData(out, y, 180)

    # from PIL import Image
    # Image.fromarray((out*255).astype(np.uint8)).show()
    # import ipdb; ipdb.set_trace(context=10)

    return out, y, rotate_angle


def test_full_augment(img):
    out = img.copy()
    out = cropRange(out, ratio=1/4)
    # out = channelScale(out)
    out = randomErase(out)
    # out = (out*255).astype(np.uint8)
    out = seq(image=out)

    return out


def create_stretched_data(img:np.ndarray, position: np.ndarray):
    aug = iaa.ScaleY(2)
    
    translated_pos = position.copy()
    min_x_value = np.min(translated_pos[:, 0])
    translated_pos[:,0] += np.abs(min_x_value)

    mask = np.round(translated_pos[:, :-1]).astype(np.uint8)
    canvas = np.zeros(img.shape, dtype=np.float32)
    for pt in mask:
        canvas[pt[0], pt[1]] = (255., 255., 255.)
    
    # canvas = canvas.astype(np.uint8)

    augmented_canvas = aug(image=canvas)
    augmented_pos = np.where(augmented_canvas==(255., 255., 255.))

    augmented_img = aug(image=img)

    return augmented_img, augmented_pos


if __name__ == '__main__':
    # import time
    # from skimage import io

    # x = io.imread('data/images/AFLW2000-crop/image00004/image00004_cropped.jpg') / 255.
    # x = x.astype(np.float32)
    # y = np.load('data/images/AFLW2000-crop/image00004/image00004_cropped_uv_posmap.npy')
    # y = y.astype(np.float32)

    # t1 = time.clock()
    # for i in range(1000):
    #     xr, yr = prnAugment_torch(x, y)

    # print(time.clock() - t1)

    import tqdm
    for i in tqdm.tqdm(range(1000)):
        img = cv2.imread("data/images/AFLW2000-crop/image00981/image00981_cropped.jpg")
        uv =  np.load("data/images/AFLW2000-crop/image00981/image00981_cropped_uv_posmap.npy", allow_pickle=True).astype(np.float32)

        img,  _, _ = prnAugment_torch(img, uv)
        cv2.imwrite(f"data/test_augment/{i}.jpg", img)
    
    # img = cv2.imread("data/images/AFLW2000-crop/image00981/image00981_cropped.jpg")
    # img = iaa.imgcorruptlike.Pixelate(severity=5)(image=img)	
    # img = img /255.
    # img, _, _ = rotateData(img, img, specify_angle=360)
    # img = (img*255).astype(np.uint8)

    # cv2.imshow('', img)
    
    # #waits for user to press any key 
    # #(this is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0) 
    
    # #closing all open windows 
    # cv2.destroyAllWindows() 
