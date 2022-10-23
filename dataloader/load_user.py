import cv2
import os
import numpy as np
import glob
from tqdm import trange
from PIL import Image


def load_user(path, scale=1, thres=0.1):
    '''
        @param
        path: images' and mask's directory. images and mask should be squared and 8-bit depth.
        scale: down sample scale, default 1.
        @return
        dict:
            images: shape(N, H//s, W//s, 3), range(0, 1), type(float32)
            mask: shape(h, w), range(0, 1), type(float32)
    '''
    images = []
    imglist = sorted(glob.glob(os.path.join(path, "*.jpg")))

    # load images
    print('loading images......')
    for i in trange(len(imglist)):
        img = np.asarray(Image.open(imglist[i]))
        img = img.astype(np.float32) / 255.
        if scale != 1:
            width = int(img.shape[1] / scale)
            height = int(img.shape[0] / scale)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
        images.append(img)
    images = np.stack(images, axis=0)
    # load mask
    mask = cv2.imread(os.path.join(path, "mask.png"), 0).astype(np.float32) / 255.
    if scale != 1:
        mask = cv2.resize(mask, dim, interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0).astype(np.float32)

    images = images * np.expand_dims(np.expand_dims(mask, axis=0), axis=-1)

    # load xy
    idx = np.where(mask > 0.5)
    xy = np.stack([idx[1], idx[0]], axis=-1).astype(np.float32)    # (p, 2)

    # normalized_xy
    normalized_xy = xy / np.array([[mask.shape[1], mask.shape[1]]])                # (p, 2), range(0,1)
    normalized_xy = normalized_xy - np.mean(normalized_xy, axis=0, keepdims=True)  # (p, 2), range(-1,1)
    normalized_xy = normalized_xy.astype(np.float32)

    # rgb
    rgb = np.stack([img[idx] for img in images], axis=0).astype(np.float32)                # (n, p, 3)

    # rgb_mean_var
    rgb_mean_var = np.concatenate([rgb.mean(0), rgb.var(0)], axis=-1).astype(np.float32)   # (p, 6)

    # shadow_map
    shadow_mask = cal_shadow_map(rgb, thres).astype(np.float32)                            # (n, p)
    return {
        'images': images,
        'mask': mask,
        'xy': xy,
        'normalized_xy': normalized_xy,
        'idx': idx,
        'rgb': rgb,
        'rgb_mean_var': rgb_mean_var,
        'shadow_mask': shadow_mask
    }

def cal_shadow_map(rgb, thres=0.1):
    '''
    Args:
        rgb: shape(n, p, 3)
        thres: default 0.1

    Returns:
        valid_shadow: shape(n, p), shadow mask
    '''
    rgb_mean = rgb.mean(axis=-1)          # (n, p)
    top_k = int(len(rgb_mean) * 0.9)      # top_k = 90% * n_light

    rgb_topk_mean = np.mean(np.sort(rgb_mean, axis=0)[:top_k], axis=0, keepdims=True)    # (1, p)
    idxp = np.where(rgb_mean >= thres * rgb_topk_mean)

    mask = np.zeros_like(rgb_mean)        # (n, p)
    mask[idxp] = 1.0
    return mask
