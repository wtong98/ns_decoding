"""
NN models.

author: William Tong
date: March 12, 2020
"""
import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras

from internal.util import sample_idxs

# TODO: implemnet early stopping
class CAE:
    def __init__(self, params):
        """
        params: {
            batch_size: size of minibatch,
            dropout_rate: rate for dropout layers
            epoch: epochs to train
            train_val_split: fraction of data to be validation data
        }
        """
        self.params = params
        self.train_ds = None
        self.test_ds = None
        self.checkpoint_path = 'save/cae_model/model.ckpt'
        self.log_path = 'save/cae_model/log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


    def load_data(self, train_path, test_path):
        train_data = np.load(train_path)
        example, label = train_data
        train_idx, val_idx = sample_idxs(example.shape[0], self.params['train_val_split'])

        self.train_ds = tf.data.Dataset.from_tensor_slices((example[train_idx], label[train_idx])) \
                                       .shuffle(256) \
                                       .batch(self.params['batch_size'])
        self.val_data = tf.data.Dataset.from_tensor_slices((example[val_idx], label[val_idx])) \
                                       .batch(self.params['batch_size'])

        test_data = np.load(test_path)
        test_example, test_label = test_data
        self.test_ds = tf.data.Dataset.from_tensor_slices((test_example, test_label)) \
                                      .batch(self.params['batch_size'])

    def load_weights(self):
        self._build_cae_nn()
        self.model.load_weights(self.checkpoint_path)

    def train(self):
        if self.train_ds is None or self.test_ds is None:
            print('Load data first!')
            return

        cp_callback = keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                      save_weights_only=True)
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.log_path, histogram_freq=1)
        es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                       patience=3,
                                                       restore_best_weights=True)


        self._build_cae_nn()
        self.model.fit(self.train_ds,
                       epochs=self.params['epoch'],
                       callbacks=[cp_callback, tb_callback, es_callback],
                       validation_data=self.val_data)

    def eval(self):
        return self.model.evaluate(self.test_ds)

    def predict(self, example=None, **kwargs):
        if example is None:
            example = self.test_ds

        return self.model.predict(example, **kwargs)


    def _build_cae_nn(self):
        model = keras.Sequential([
            # encoder
            keras.layers.Conv2D(filters=64, kernel_size=7, strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),

            keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.params['dropout_rate']),

            keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.params['dropout_rate']),

            keras.layers.Conv2D(filters=256, kernel_size=3, strides=(2,2), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.params['dropout_rate']),

            # decoder
            keras.layers.UpSampling2D(size=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=3, strides=(1,1), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.params['dropout_rate']),

            keras.layers.UpSampling2D(size=(2,2)),
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=(1,1), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.params['dropout_rate']),

            keras.layers.UpSampling2D(size=(2,2)),
            keras.layers.Conv2D(filters=64, kernel_size=5, strides=(1,1), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Dropout(self.params['dropout_rate']),

            keras.layers.UpSampling2D(size=(2,2)),
            keras.layers.Conv2D(filters=1, kernel_size=7, strides=(1,1), padding='same'),
            keras.layers.BatchNormalization(),

            # adjust for upsampling distortion
            keras.layers.Cropping2D(((1,0), (7, 7)))
        ])

        model.compile(optimizer='adam',
                      loss='mse')

        self.model = model