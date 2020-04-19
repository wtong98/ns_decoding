"""
NN models.

author: William Tong
date: March 12, 2020
"""
import datetime
import os
import os.path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from internal.util import sample_idxs

# TODO: check image normalization
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
        self.log_path = 'save/cae_model/log/' + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


    def load_test(self, test_path):
        test_data = np.load(test_path) / 255
        test_example, test_label = test_data
        self.test_ds = tf.data.Dataset.from_tensor_slices((test_example, test_label)) \
                                      .batch(self.params['batch_size'])

    def load_images(self, truth_dir, blur_dir):
        # im_names = os.listdir(truth_dir)
        # name_ds = tf.data.Dataset.from_tensor_slices(im_names)
        # num_shards = int(1 / self.params['train_val_split'])
        #
        # name_val_ds = name_ds.shard(num_shards, 0)
        # name_train_ds = name_ds.shard(num_shards, 1)
        # for i in range(2, num_shards):
        #     name_train_ds = name_train_ds.concatenate(name_ds.shard(num_shards, i))

        im_names = os.listdir(truth_dir)
        def read_path(image_path):
            img = np.load(image_path) / 255
            return np.expand_dims(img, -1)

        def assemble(image_name):
            truth_im = read_path(os.path.join(truth_dir, image_name))
            blur_im = read_path(os.path.join(blur_dir, image_name))
            return (blur_im, truth_im)

        def image_gen():
            for name in im_names:
                yield assemble(name)

        total_ds = tf.data.Dataset.from_generator(image_gen, (tf.float64, tf.float64),
                                output_shapes=(tf.TensorShape((None, None, None)), tf.TensorShape((None, None, None))))
        num_shards = int(1 / self.params['train_val_split'])

        # TODO: avoid sharding for validation
        self.val_ds = total_ds.shard(num_shards, 0)
        self.train_ds = total_ds.shard(num_shards, 1)
        for i in range(2, num_shards):
            self.train_ds = self.train_ds.concatenate(total_ds.shard(num_shards, i))

        self.val_ds = self.val_ds.batch(self.params['batch_size'])
        self.train_ds = self.train_ds.shuffle(256) \
                                     .batch(self.params['batch_size'])

        # AUTOTUNE = tf.data.experimental.AUTOTUNE
        # self.val_ds = name_val_ds.map(assemble, num_parallel_calls=AUTOTUNE) \
        #                          .batch(self.params['batch_size'])
        #
        # self.train_ds = name_train_ds.map(assemble, num_parallel_calls=AUTOTUNE) \
        #                              .shuffle(256) \
        #                              .batch(self.params['batch_size'])

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
                       validation_data=self.val_ds)

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
