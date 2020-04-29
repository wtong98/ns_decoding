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
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from tqdm import tqdm

from internal.util import sample_idxs

IMAGE_DIMS = (95, 146)

# <codecell>
train, test = sample_idxs(20000, 0.1)

# <codecell>
images = np.load('save/test_images.npy')
smoothed_images = np.load('save/smooth_test_images.npy')
decoded_images = np.load('save/smooth_test_decoded.npy')
residuals = smoothed_images - decoded_images


# <codecell>
def build_dataset(n_samps_per_image=10, train_test_split=0.2):
    train_idxs, test_idxs = sample_idxs(images.shape[0], train_test_split)

    train_dataset = np.array(list(zip(*_dataset_gen(n_samps_per_image, train_idxs))))
    test_dataset = np.stack([decoded_images[test_idxs].reshape(-1, *IMAGE_DIMS),
                             images[test_idxs].reshape(-1, *IMAGE_DIMS)], axis=0)

    train_dataset = np.expand_dims(train_dataset, -1)
    test_dataset = np.expand_dims(test_dataset, -1)

    return (train_dataset, test_dataset)

def _dataset_gen(n_samps, train_idxs):
    for i in tqdm(range(len(train_idxs))):
        for _ in range(n_samps):
            res = residuals[i,:].reshape(IMAGE_DIMS)
            target_image = images[i,:].reshape(IMAGE_DIMS)
            samp_residual = sample_residual(res)
            samp_image = target_image - samp_residual

            yield (samp_image, target_image)

def build_dataset_gpr(n_samps_per_image=10, train_test_split=0.2, gpr_train_size=5000):
    train_idxs, test_idxs = sample_idxs(images.shape[0], train_test_split)

    train_dataset = np.array(list(zip(*_dataset_gen_gpr(n_samps_per_image, train_idxs, gpr_train_size))))
    test_dataset = np.stack([decoded_images[test_idxs].reshape(-1, *IMAGE_DIMS),
                             images[test_idxs].reshape(-1, *IMAGE_DIMS)], axis=0)

    train_dataset = np.expand_dims(train_dataset, -1)
    test_dataset = np.expand_dims(test_dataset, -1)

    return (train_dataset, test_dataset)

def _dataset_gen_gpr(n_samps, train_idxs, gpr_train_size):
    preds, _ = fit_gpr_resid_model(gpr_train_size,
                                   decoded_images[train_idxs],
                                   residuals[train_idxs])
    pred_res = generate_pred_residuals(decoded_images[train_idxs], preds)

    for i in tqdm(range(len(train_idxs))):
        for _ in range(n_samps):
            res = residuals[i,:].reshape(IMAGE_DIMS)
            curr_pred_res = pred_res[i,:,:]
            target_image = images[i,:].reshape(IMAGE_DIMS)
            samp_residual = sample_residual(res, curr_pred_res) + curr_pred_res
            samp_image = target_image - samp_residual

            yield (samp_image, target_image)

# TODO: sample GP?
def fit_gpr_resid_model(train_size, decoded_images, residuals, **kwargs):
    idxs = random.sample(range(decoded_images.shape[0] * decoded_images.shape[1]), train_size)
    regressor = GaussianProcessRegressor(**kwargs)
    X = np.concatenate(decoded_images, axis=0).reshape(-1, 1) / 255
    Y = np.concatenate(residuals, axis=0).reshape(-1, 1) / 255
    regressor.fit(X[idxs], Y[idxs])
    preds, sd = regressor.predict(np.linspace(0, 1, 256).reshape(-1, 1), return_std=True)

    return preds * 255, sd * 255

def generate_pred_residuals(decoded_images, gpr_preds):
    im_idxs = np.floor(decoded_images.flatten()).astype("int")
    im_idxs[im_idxs > 255] = 255
    im_idxs[im_idxs < 0] = 0

    pred_res = gpr_preds[im_idxs].reshape(-1, *IMAGE_DIMS)
    return pred_res

# def fit_gpr_resid_model(train_size, decoded_images, residuals, **kwargs):
#     idxs = random.sample(range(decoded_images.shape[0] * decoded_images.shape[1]), train_size)
#     regressor = GaussianProcessRegressor(**kwargs)
#     X = np.concatenate(decoded_images, axis=0).reshape(-1, 1) / 255
#     Y = np.concatenate(residuals, axis=0).reshape(-1, 1) / 255
#     regressor.fit(X[idxs], Y[idxs])
#
#     return regressor
#
# def generate_pred_residuals(decoded_images, gpr_reg):
#     all_res = []
#     for im in tqdm(decoded_images.reshape(-1, *IMAGE_DIMS)):
#         im_slices = []
#         print(im.shape)
#         for slice in im:
#             preds = gpr_reg.sample_y(slice.reshape(-1, 1)).flatten()
#             im_slices.append(preds)
#
#         pred_res = np.stack(im_slices, axis=0)
#         all_res.append(pred_res)
#
#     return np.stack(all_res, axis=0)

def sample_residual(residual: np.ndarray,
                        pred_res: np.ndarray = np.zeros(IMAGE_DIMS)) -> np.ndarray:
    res_norm = np.linalg.norm(residual, ord='fro')
    pred_res_norm = np.linalg.norm(pred_res, ord='fro')
    diff = np.sqrt(max(0, res_norm ** 2 - pred_res_norm ** 2))

    noise = _make_smooth_gaussian_noise(IMAGE_DIMS)
    noise_norm = np.linalg.norm(noise, ord='fro')
    sample = (diff / noise_norm) * noise

    # print('res_norm', res_norm)
    # print('pred_res_norm', pred_res_norm)
    # print('diff', diff)
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
            # samp_residual = sample_residual(residual, pred_res[i,:,:]) + pred_res[i,:,:]
            # print('var_res', np.linalg.norm(residual, ord='fro'))
            # print('var_samp', np.linalg.norm(samp_residual, ord='fro'))
            curr_pred_res = pred_res[i,:,:]

            fig = _plot_row(i, truth, blur, decoded + curr_pred_res, residual - curr_pred_res, samp_residual)
            pdf.savefig(fig)


# def _plot_row(image_no, blur, decoded, residual, pred_res, sample_res):
#     res_normalize = Normalize(vmin=np.min(sample_res), vmax=np.max(sample_res))
#     fig, axs = plt.subplots(ncols=6, figsize=(15,5))
#
#     # im1 = axs[0].imshow(truth, cmap='Greys_r')
#     # axs[0].set_title("Ground truth: "+"Image "+str(image_no))
#     # axs[0].axis('off')
#
#     axs[0].imshow(blur, cmap='Greys_r')
#     axs[0].set_title("Smoothed: "+"Image "+str(image_no))
#     axs[0].axis('off')
#
#     im1 = axs[1].imshow(decoded, cmap='Greys_r')
#     axs[1].set_title("Decoded: "+"Image "+str(image_no))
#     axs[1].axis('off')
#
#     axs[2].imshow(residual, cmap='Reds_r')
#     axs[2].set_title("Residuals: "+"Image "+str(image_no))
#     axs[2].axis('off')
#
#     axs[3].imshow(pred_res, cmap='Reds_r', norm=res_normalize)
#     axs[3].set_title("GP Predicted Res: "+"Image "+str(image_no))
#     axs[3].axis('off')
#
#     im5 = axs[4].imshow(sample_res, cmap='Reds_r')
#     axs[4].set_title("Sampled residuals: "+"Image "+str(image_no))
#     axs[4].axis('off')
#
#     axs[5].axis('off')
#
#     fig.tight_layout()
#     fig.colorbar(im1, shrink=0.4, pad=0.5)
#     fig.colorbar(im5, shrink=0.4, pad=0.5)

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

def compute_pixelwise_corr(first, second):
    corr_mat = np.zeros(IMAGE_DIMS)
    for i, slice in tqdm(enumerate(corr_mat)):
        for j, _ in enumerate(slice):
            corr_mat[i,j] = np.corrcoef(first[:,i,j], second[:,i,j])[0][1]

    return corr_mat


#################
# HYDROGEN_WORK #
#################

# <codecell>
# val_idx, test_idx = sample_idxs(images.shape[0], 0.25)
# val_data = decoded_images[val_idx]
# test_data = decoded_images[test_idx]
#
# # <codecell>
# means = []
# for i in tqdm(range(256)):
#     idxs = np.floor(val_data.flat) == i
#     mean = np.mean((residuals[val_idx].flatten())[idxs])
#     means.append(mean)
#
# # <codecell>
# preds, sd = fit_gpr_resid_model(5000, val_data, residuals[val_idx], alpha=5e-5)
# pred_res = generate_pred_residuals(test_data, preds)
#
# # <codecell>
# preds = preds.flatten()
# plt.plot(range(256), means, label="Sample mean")
# plt.plot(range(256), preds, label="GP fit")
# plt.fill_between(range(256), preds - sd, preds + sd, alpha=0.2, color='orange')
# plt.title("GP Fit with RBF + White noise kernel")
# plt.xlabel("Pixel value")
# plt.ylabel("Residual")
# plt.legend()
# plt.savefig('save/gp_fit.png', dpi=150)
#
#
# # <codecell>
# make_pdf_plot(20, range(100), pred_res, 'save/')
#
# # <codecell>
# gp_decode = test_data.reshape(-1, *IMAGE_DIMS) + pred_res
# corr_gp = compute_pixelwise_corr(gp_decode,
#                                  smoothed_images[test_idx].reshape(-1, *IMAGE_DIMS))
# corr_test = compute_pixelwise_corr(test_data.reshape(-1, *IMAGE_DIMS),
#                                    smoothed_images[test_idx].reshape(-1, *IMAGE_DIMS))
#
# # <codecell>
# print("mean gp corr", np.mean(corr_gp))
# print("mean test corr", np.mean(corr_test))

###############
# SCRIPT_WORK #
###############

# <codecell>
train_ds, test_ds = build_dataset()
np.save('save/cae_train_dataset.npy', train_ds)
np.save('save/cae_test_dataset.npy', test_ds)


















# <codecell>
