"""
Exploratory script for sampling decoded images L

authors:
    William Tong (wlt2115@columbia.edu)
    Young Joon Kim (yk2611@columbia.edu)
date: March 5, 2020
"""

# <codecell>
import os.path
import random

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

IMAGE_DIMS = (95, 146)

# <codecell>
images = np.load('save/test_images.npy')
smoothed_images = np.load('save/smooth_test_images.npy')
decoded_images = np.load('save/smooth_test_decoded.npy')
residuals = smoothed_images - decoded_images

# <codecell>

def sample_residual(residual: np.ndarray) -> np.ndarray:
    res_norm = np.linalg.norm(residual, ord='fro')

    noise = _make_smooth_gaussian_noise(IMAGE_DIMS)
    noise_norm = np.linalg.norm(noise, ord='fro')
    sample = (res_norm / noise_norm) * noise

    return sample


def _make_smooth_gaussian_noise(dimension: tuple) -> np.ndarray:
    noise = np.random.normal(size=dimension)
    smoothed_noise = gaussian_filter(noise,
                                     sigma=4,
                                     order=0,
                                     mode='constant',
                                     cval=0.0,
                                     truncate=3.0)
    return smoothed_noise



def make_pdf_plot(num_images, save_dir):
    with PdfPages(os.path.join(save_dir, "sample_residual_comparison.pdf")) as pdf:
        idxs = random.sample(range(100), num_images)
        for i in tqdm(idxs):
            truth = images[i,:].reshape(IMAGE_DIMS)
            blur = smoothed_images[i,:].reshape(IMAGE_DIMS)
            residual = residuals[i,:].reshape(IMAGE_DIMS)
            samp_residual = sample_residual(residual)

            fig = _plot_row(i, truth, blur, residual, samp_residual)
            pdf.savefig(fig)


def _plot_row(image_no, truth, blur, residual, sample_res):
    fig, axs = plt.subplots(ncols=4, figsize=(15,5))

    stim = axs[0].imshow(truth, cmap='Greys_r')
    axs[0].set_title("Ground truth: "+"Image "+str(image_no))
    axs[0].axis('off')

    yass = axs[1].imshow(blur, cmap='Greys_r')
    axs[1].set_title("Smoothed: "+"Image "+str(image_no))
    axs[1].axis('off')

    ks2 = axs[2].imshow(residual, cmap='Greys_r')
    axs[2].set_title("Residuals: "+"Image "+str(image_no))
    axs[2].axis('off')

    unsorted = axs[3].imshow(sample_res, cmap='Greys_r')
    axs[3].set_title("Sampled residuals: "+"Image "+str(image_no))
    axs[3].axis('off')

    fig.tight_layout()



# <codecell>
make_pdf_plot(20, 'save/')


















# <codecell>
