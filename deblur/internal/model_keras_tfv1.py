"""
NN models.

author: William Tong
date: March 12, 2020
"""
import datetime
import math

import h5py
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
        self.train_ds = self.train_steps = None
        self.val_ds = self.val_steps = None
        self.test_ds = self.test_steps = None
        self.checkpoint_path = 'save/cae_model/model.ckpt'
        self.log_path = 'save/cae_model/log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.h5fp = None

        self._build_cae_nn()

    def load_data(self, path):
        example = label = test_example = test_label = None
        fp = self.h5fp = h5py.File(path, 'r')
        batch_size = self.params['batch_size']

        example = fp['train_blur']
        label = fp['train_truth']
        self.train_steps = math.ceil(len(example) / batch_size)

        val_example = fp['val_blur']
        val_label = fp['val_truth']
        self.val_steps = math.ceil(len(val_example) / batch_size)

        test_example = fp['test_blur']
        test_label = fp['test_truth']
        self.test_steps = math.ceil(len(test_example) / batch_size)

        self.train_ds = tf.data.Dataset.from_tensor_slices((example, label)) \
                                       .shuffle(256) \
                                       .batch(batch_size)
        self.val_ds = tf.data.Dataset.from_tensor_slices((val_example, val_label)) \
                                       .batch(batch_size)

        self.test_ds = tf.data.Dataset.from_tensor_slices((test_example, test_label)) \
                                      .batch(batch_size)

    # def load_data(self, path):
    #     self.h5fp = h5py.File(path, 'r')
    #
    #     def build_batched_gen(blur_name, truth_name, batch_size, chunk_factor = 10):
    #         fp = self.h5fp
    #         data_len = len(fp[blur_name])
    #         chunk_size = batch_size * chunk_factor
    #         total_chunks = math.ceil(data_len / chunk_size)
    #
    #         def generator():
    #             while True:
    #                 for chunk_idx in range(total_chunks):
    #                     start = chunk_idx * chunk_size
    #                     end = min(start + chunk_size, data_len)
    #                     rand_idx, _ = sample_idxs(end - start, 0)
    #
    #                     blur_chunk = fp[blur_name][start:end][rand_idx]
    #                     truth_chunk = fp[truth_name][start:end][rand_idx]
    #
    #                     chunk_len = len(blur_chunk)
    #                     total_batches = math.ceil(chunk_len / batch_size)
    #                     for batch_idx in range(total_batches):
    #                         start = batch_idx * batch_size
    #                         end = min(start + batch_size, chunk_len)
    #
    #                         blur_batch = blur_chunk[start:end]
    #                         truth_batch = truth_chunk[start:end]
    #                         yield blur_batch, truth_batch
    #
    #         num_batches = math.ceil(data_len / batch_size)
    #         return generator(), num_batches
    #
    #
    #     self.train_ds, self.train_steps = build_batched_gen('train_blur', 'train_truth', batch_size=32)
    #     self.val_ds, self.val_steps = build_batched_gen('val_blur', 'val_truth', batch_size=32)
    #     self.test_ds, self.test_steps = build_batched_gen('test_blur', 'test_truth', batch_size=32)

    def load_weights(self):
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


        self.model.fit(self.train_ds,
                       epochs=self.params['epoch'],
                       callbacks=[cp_callback, tb_callback, es_callback],
                       validation_data=self.val_ds,
                       steps_per_epoch=self.train_steps,
                       validation_steps=self.val_steps)

    def eval(self):
        return self.model.evaluate(self.test_ds)

    def predict(self):
        return self.model.predict(self.test_ds, steps=self.test_steps)


    def close():
        if self.h5fp is not None:
            self.h5fp.close()


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
