"""
Deblurs decoded images using a CAE.

author: William Tong
date: March 13, 2020
"""

# <codecell>
"""
NN models.

author: William Tong
date: March 12, 2020
"""

# <codecell>
import os.path
import random

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from internal.model_tfv1 import SRGAN

# <codecell>(
params = {
    'batch_size': 32,
    'dropout_rate': 0.5,
    'epoch': 2,
    'train_val_split': 0.1,
    'total_images': 391,
    'check_train_every': 1,
    'check_val_every': 2,
    'early_stopping': True,
    'stale_after': 5,
    'early_stop_ratio': 0.99
}

model = SRGAN(params)
model.load_data(path=r'save/cae_dataset.h5py')

# <codecell>
model.train()

# <codecell>
preds = model.predict()
np.save('save/cae_preds.npy', preds)

# <codecell>

###############
# PLOT_MAKING #
###############
preds = np.squeeze(np.load('save/cae_preds_gpu.npy'))
test_blur, test_target = np.squeeze(np.load('save/cae_dataset_test_gpu.npy'))

preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))
# fp = h5py.File('save/cae_dataset.h5py', 'r')
# test_blur = np.squeeze(fp['test_blur'][:])
# test_target = np.squeeze(fp['test_truth'][:])
# fp.close()

# <codecell>
def make_pdf_plot(num_images, save_dir):
    with PdfPages(os.path.join(save_dir, "deblur_cae.pdf")) as pdf:
        idxs = random.sample(range(test_blur.shape[0]), num_images)
        idxs.sort()
        for i in tqdm(idxs):
            truth = test_target[i,:,:]
            blur = test_blur[i,:,:]
            cae = preds[i,:,:]

            fig = _plot_row(i, truth, blur, cae)
            pdf.savefig(fig)


def _plot_row(image_no, truth, blur, cae):
    fig, axs = plt.subplots(ncols=3, figsize=(15,5))

    im1 = axs[0].imshow(truth, cmap='Greys_r')
    axs[0].set_title("Ground truth: "+"Image "+str(image_no))
    axs[0].axis('off')

    im2 = axs[1].imshow(blur, cmap='Greys_r')
    axs[1].set_title("Blurred: "+"Image "+str(image_no))
    axs[1].axis('off')

    im3 = axs[2].imshow(cae, cmap='Greys_r')
    axs[2].set_title("CAE denoised: "+"Image "+str(image_no))
    axs[2].axis('off')
    fig.tight_layout()


# <codecell>
make_pdf_plot(20, 'save/')

# <codecell>

mse_before_cae = np.mean(np.square(test_blur - test_target), axis=(1,2))
mse_after_cae = np.mean(np.square(preds - test_target), axis=(1,2))

one_to_one_line = np.linspace(0, 0.1, 100)
plt.scatter(mse_before_cae, mse_after_cae)
plt.plot(one_to_one_line, one_to_one_line, 'r--')
plt.title('MSE Before and After CAE')
plt.xlabel('Before')
plt.ylabel('After')
plt.savefig('save/mse_before_and_after_cae.png', dpi=150)


# <codecell>
mse_before_cae.shape
np.mean(mse_after_cae)





# <codecell>
