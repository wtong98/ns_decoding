"""
NN models.

author: William Tong
date: March 12, 2020
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

class CAE:
    def __init__(self, params):
        """
        params: {
            batch_size: size of minibatch,
            dropout_rate: rate for dropout layers
            epoch: epochs to train
        }
        """
        self.params = params
        self.train_ds = None
        self.test_ds = None
        self.checkpoint_path = 'save/cae_model/model.ckpt'


    def load_data(self, train_path, test_path):
        train_data = np.load(train_path)
        train_example, train_label = train_data
        self.train_ds = tf.data.Dataset.from_tensor_slices((train_example, train_label)) \
                                       .shuffle(256) \
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

        self._build_cae_nn()
        self.model.fit(self.train_ds,
                       epochs=self.params['epoch'],
                       callbacks=[cp_callback])

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
                      loss='mse',
                      metrics=['mse'])

        self.model = model
