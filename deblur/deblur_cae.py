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
import numpy as np

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
