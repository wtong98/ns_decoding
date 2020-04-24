"""
Script for preprocessing imagenet images. Builds on sample.py

author: William Tong (wlt2115@columbia.edu)
date: April 16, 2020
"""

# <codecell>
import os
import os.path
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter
from skimage import io, transform, util
from tqdm import tqdm

from internal.util import sample_idxs

IMAGE_DIMS = (95, 146)
GF_ARGS = {
    "sigma": 4,
    "order": 0,
    "mode": 'constant',
    "cval": 0.0,
    "truncate": 3.0,
}

# <codecell>
def process_image(image_dir):
    smoothed_images = np.load('save/smooth_test_images.npy')
    decoded_images = np.load('save/smooth_test_decoded.npy')
    residuals = _normalize(smoothed_images - decoded_images)
    variance = np.mean(np.var(residuals, axis=0))

    image_paths = _ls_images(image_dir)
    image_arr = _read_image(image_paths)
    trim_im = _stack(_trim_image(image_arr), save_name="train_truth")
    samp_im = _stack(_sample(trim_im, variance), save_name="train_blur")

    total = sum([1 for _ in samp_im])
    print('Processed %d images' % total)

def _ls_images(image_dir):
    for root, _, files in os.walk(image_dir):
        for name in files:
            yield os.path.join(root, name)

def _read_image(image_paths):
    for path in image_paths:
        yield io.imread(path, as_gray=True)

def _trim_image(images, save_path=None):
    vals = []
    for im in images:
        trim_im = _center_crop(_adjust_size(im))
        yield trim_im

def _adjust_size(image):
    x_adj = IMAGE_DIMS[0] / image.shape[0]
    y_adj = IMAGE_DIMS[1] / image.shape[1]
    scale_factor = max(x_adj, y_adj)

    return transform.rescale(image, scale_factor)

def _center_crop(image):
    x_start = int((image.shape[0] - IMAGE_DIMS[0]) / 2)
    y_start = int((image.shape[1] - IMAGE_DIMS[1]) / 2)
    x_end = image.shape[0] - (IMAGE_DIMS[0] + x_start)
    y_end = image.shape[1] - (IMAGE_DIMS[1] + y_start)

    return util.crop(image, ((x_start, x_end), (y_start, y_end)), copy=True)

def _sample(images, variance):
    for im in images:
        blur_im = gaussian_filter(im, **GF_ARGS)
        samp_res = sample_residual(variance, **GF_ARGS)
        yield blur_im - samp_res

def _stack(image_gen, save_name, chunk_size=100):
    vals = []
    chunk_idx = 0
    for im in tqdm(image_gen):
        vals.append(im)

        if len(vals) >= chunk_size:
            data = _normalize(np.stack(vals, axis=0))
            _save_to_ds(save_name,
                        np.expand_dims(data, -1),
                        chunk_size=chunk_size,
                        chunk_idx=chunk_idx)
            chunk_idx += 1
            vals.clear()

        yield im

    if len(vals) >= 0:
        data = _normalize(np.stack(vals, axis=0))
        _save_to_ds(save_name,
                    np.expand_dims(data, -1),
                    chunk_size=chunk_size,
                    chunk_idx=chunk_idx,
                    final_chunk=True)



def _save_to_ds(save_name, arr, save_path='save/cae_dataset.h5py',
                chunk_size=None, chunk_idx=0, final_chunk=False):
    with h5py.File(save_path, 'a') as fp:
        if chunk_size is None:
            ds = fp.create_dataset(save_name, arr.shape)
            ds[:] = arr
        else:
            data_shape = list(arr.shape[1:])
            ds = fp[save_name] if save_name in fp \
                else fp.create_dataset(save_name,
                                       shape=(chunk_size * 10, *data_shape),
                                       chunks=(chunk_size, *data_shape),
                                       maxshape=(None, *data_shape))
            start = chunk_idx * chunk_size
            end = start + min((chunk_idx + 1) * chunk_size, arr.shape[0])

            if end >= ds.shape[0]:
                ds.resize(int(ds.shape[0] * 1.5), axis=0)

            ds[start:end] = arr

            if final_chunk:
                ds.resize(end, axis=0)


def _normalize(arr):
    min = np.min(arr)
    range = np.max(arr) - min
    return (arr - min) / range


def sample_residual(variance: float, **filter_args) -> np.ndarray:
    noise = _make_smooth_gaussian_noise(IMAGE_DIMS, **filter_args)
    noise_norm = np.linalg.norm(noise, ord='fro')
    sample = (np.sqrt(variance) / noise_norm) * noise

    return sample


def _make_smooth_gaussian_noise(dimension: tuple, **filter_args) -> np.ndarray:
    noise = np.random.normal(size=dimension)
    smoothed_noise = gaussian_filter(noise, **filter_args)
    return smoothed_noise


# <codecell>
process_image(r'/home/grandpaa/workspace/neural_decoding/dataset/imagenet_images')

# <codecell>
IMAGE_DIMS = (95, 146)
images = _normalize(np.load('save/test_images.npy'))
decoded_images = _normalize(np.load('save/smooth_test_decoded.npy'))
test_data = np.stack((decoded_images.reshape(-1, *IMAGE_DIMS),
                      images.reshape(-1, *IMAGE_DIMS)), axis=0)

# TODo: incorporate into model and deblur_cae
_save_to_ds("test_blur", decoded_images.reshape(-1, *IMAGE_DIMS, 1))
_save_to_ds("test_truth", images.reshape(-1, *IMAGE_DIMS, 1))
