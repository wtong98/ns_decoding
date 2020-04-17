"""
Script for preprocessing imagenet images. Builds on sample.py

author: William Tong (wlt2115@columbia.edu)
date: April 16, 2020
"""

# <codecell>
import os
import os.path
import random

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
smoothed_images = np.load('save/smooth_test_images.npy')
decoded_images = np.load('save/smooth_test_decoded.npy')
residuals = (smoothed_images - decoded_images) / 255

# <codecell>
def process_image(image_dir):
    variance = np.mean(np.var(residuals, axis=0))
    print('variance:', variance)

    image_paths = _ls_images(image_dir)
    image_arr = _read_image(image_paths)
    trim_im = _save_image(_trim_image(image_arr), r'save/images/truth')
    samp_im = _save_image(_sample(trim_im, variance), r'save/images/blur')

    total = sum([1 for _ in samp_im])
    print('Processed %d images' % total)

def _ls_images(image_dir):
    for root, _, files in os.walk(image_dir):
        for name in files:
            yield os.path.join(root, name)

def _read_image(image_paths):
    for path in image_paths:
        yield io.imread(path, as_gray=True)

def _trim_image(images):
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

def _save_image(images, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o755)

    count = 0
    for im in tqdm(images):
        path = os.path.join(save_dir, "im_%d.jpg" % count)
        io.imsave(path, _to_byte(im))
        count += 1

        yield im

def _to_byte(image):
    min_val = np.min(image)
    range_val = np.max(image) - min_val
    byte_image = (255 / range_val * (image - min_val)).astype('uint8')

    return byte_image


def sample_residual(variance: float, **filter_args) -> np.ndarray:
    noise = _make_smooth_gaussian_noise(IMAGE_DIMS, **filter_args)
    noise_norm = np.linalg.norm(noise, ord='fro')
    sample = (np.sqrt(variance) / noise_norm) * noise

    return sample


def _make_smooth_gaussian_noise(dimension: tuple, **filter_args) -> np.ndarray:
    noise = np.random.normal(size=dimension)
    smoothed_noise = gaussian_filter(noise, **filter_args)
    return smoothed_noise


def make_pdf_plot(num_images, train_idxs, pred_res, save_dir):
    with PdfPages(os.path.join(save_dir, "sample_residual_comparison.pdf")) as pdf:
        idxs = random.sample(list(train_idxs), num_images)
        idxs.sort()
        for i in tqdm(idxs):
            truth = images[i,:].reshape(IMAGE_DIMS)
            blur = smoothed_images[i,:].reshape(IMAGE_DIMS)
            decoded = decoded_images[i,:].reshape(IMAGE_DIMS)
            residual = residuals[i,:].reshape(IMAGE_DIMS)
            samp_residual = sample_residual(residual)
            curr_pred_res = pred_res[i,:,:]

            fig = _plot_row(i, truth, blur, decoded + curr_pred_res, residual - curr_pred_res, samp_residual)
            pdf.savefig(fig)


def _plot_row(image_no, truth, blur, decoded_gpr, residual, sample_res):
    fig, axs = plt.subplots(ncols=6, figsize=(15,5))

    im1 = axs[0].imshow(truth, cmap='Greys_r')
    axs[0].set_title("Ground truth: "+"Image "+str(image_no))
    axs[0].axis('off')

    axs[1].imshow(blur, cmap='Greys_r')
    axs[1].set_title("Smoothed: "+"Image "+str(image_no))
    axs[1].axis('off')

    im1 = axs[2].imshow(decoded_gpr, cmap='Greys_r')
    axs[2].set_title("Decoded + GP: "+"Image "+str(image_no))
    axs[2].axis('off')

    axs[3].imshow(residual, cmap='Reds_r')
    axs[3].set_title("Residuals: "+"Image "+str(image_no))
    axs[3].axis('off')

    im5 = axs[4].imshow(sample_res, cmap='Reds_r')
    axs[4].set_title("Sampled residuals: "+"Image "+str(image_no))
    axs[4].axis('off')

    axs[5].axis('off')

    fig.tight_layout()
    fig.colorbar(im1, shrink=0.4, pad=0.5)
    fig.colorbar(im5, shrink=0.4, pad=0.5)


# <codecell>
process_image(r'/home/grandpaa/workspace/neural_decoding/dataset/imagenet_images')
