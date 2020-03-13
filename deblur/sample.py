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
def build_dataset(n_samps_per_image=10, train_test_split=0.2):
    total_len = images.shape[0]
    test_len = int(train_test_split * total_len)
    idxs = random.sample(range(total_len), total_len)
    test_idxs = idxs[:test_len]
    train_idxs = idxs[test_len:]

    train_dataset = np.array(list(zip(*_dataset_gen(n_samps_per_image, train_idxs))))
    test_dataset = np.stack([decoded_images[test_idxs].reshape(-1, *IMAGE_DIMS),
                             images[test_idxs].reshape(-1, *IMAGE_DIMS)], axis=0)

    train_dataset = np.expand_dims(train_dataset, -1)
    test_dataset = np.expand_dims(test_dataset, -1)

    return (train_dataset, test_dataset)

def _dataset_gen(n_samps, train_idxs):
    for i in tqdm(range(len(train_idxs))):
        for _ in range(n_samps):
            samp_residual = sample_residual(residuals[i,:].reshape(IMAGE_DIMS))
            target_image = images[i,:].reshape(IMAGE_DIMS)
            samp_image = target_image - samp_residual

            yield (samp_image, target_image)

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
        idxs.sort()
        for i in tqdm(idxs):
            truth = images[i,:].reshape(IMAGE_DIMS)
            blur = smoothed_images[i,:].reshape(IMAGE_DIMS)
            decoded = decoded_images[i,:].reshape(IMAGE_DIMS)
            residual = residuals[i,:].reshape(IMAGE_DIMS)
            samp_residual = sample_residual(residual)

            fig = _plot_row(i, truth, blur, decoded, residual, samp_residual)
            pdf.savefig(fig)


def _plot_row(image_no, truth, blur, decoded, residual, sample_res):
    fig, axs = plt.subplots(ncols=6, figsize=(15,5))

    im1 = axs[0].imshow(truth, cmap='Greys_r')
    axs[0].set_title("Ground truth: "+"Image "+str(image_no))
    axs[0].axis('off')

    im2 = axs[1].imshow(blur, cmap='Greys_r')
    axs[1].set_title("Smoothed: "+"Image "+str(image_no))
    axs[1].axis('off')

    im3 = axs[2].imshow(decoded, cmap='Greys_r')
    axs[2].set_title("Decoded: "+"Image "+str(image_no))
    axs[2].axis('off')

    im4 = axs[3].imshow(residual, cmap='Reds_r')
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
make_pdf_plot(20, 'save/')

# <codecell>
train_ds, test_ds = build_dataset()
np.save('save/cae_train_dataset.npy', train_ds)
np.save('save/cae_test_dataset.npy', test_ds)


















# <codecell>
