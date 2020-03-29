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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from internal.model import CAE

# <codecell>
params = {
    'batch_size': 32,
    'dropout_rate': 0.5,
    'epoch': 1
}

model = CAE(params)
model.load_data(train_path='save/cae_train_dataset.npy',
                test_path='save/cae_test_dataset.npy')

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
test_blur, test_target = np.squeeze(np.load('save/cae_test_dataset_gpu.npy'))

# <codecell>
def make_pdf_plot(num_images, save_dir):
    with PdfPages(os.path.join(save_dir, "deblur_cae.pdf")) as pdf:
        idxs = random.sample(range(200), num_images)
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
# make_pdf_plot(20, 'save/')

mse_before_cae = np.mean(np.square(test_blur - test_target), axis=(1,2))
mse_after_cae = np.mean(np.square(preds - test_target), axis=(1,2))

one_to_one_line = np.arange(0, 5000)
plt.scatter(mse_before_cae, mse_after_cae)
plt.plot(one_to_one_line, one_to_one_line, 'r--')
plt.title('MSE Before and After CAE')
plt.xlabel('Before')
plt.ylabel('After')
plt.savefig('save/mse_before_and_after_cae.png', dpi=150)


# <codecell>
mse_before_cae.shape






# <codecell>
