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
        self.train_ds = None
        self.test_ds = None
        self.save_path = 'save/cae_model/model.ckpt'
        self.h5py = None
        # self.log_path = 'save/cae_model/log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        input_shape = (None, 95, 146, 1)
        self.input_x_ph = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_x')
        self.input_y_ph = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_y')
        self.phase_train_ph = tf.placeholder(tf.bool, name='phase_train_ph')

        self._build_cae_nn()
        self.loss = tf.reduce_mean(tf.squared_difference(self.model, self.input_y_ph))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0004).minimize(self.loss)
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=3)


    def load_data(self, path):
        self.h5fp = h5py.File(path, 'r')

        def batched_gen(blur_name, truth_name, batch_size, chunk_factor = 10, shuffle=False):
            fp = self.h5fp
            data_len = len(fp[blur_name])
            chunk_size = batch_size * chunk_factor
            total_chunks = math.ceil(data_len / chunk_size)

            for chunk_idx in range(total_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, data_len)

                blur_chunk = fp[blur_name][start:end]
                truth_chunk = fp[truth_name][start:end]

                if shuffle:
                    rand_idx, _ = sample_idxs(end - start, 0)
                    blur_chunk = blur_chunk[rand_idx]
                    truth_chunk = truth_chunk[rand_idx]

                chunk_len = len(blur_chunk)
                total_batches = math.ceil(chunk_len / batch_size)
                for batch_idx in range(total_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, chunk_len)

                    blur_batch = blur_chunk[start:end]
                    truth_batch = truth_chunk[start:end]
                    yield blur_batch, truth_batch

        self.train_ds = lambda: batched_gen('train_blur', 'train_truth', batch_size=32, shuffle=True)
        self.val_ds = lambda: batched_gen('val_blur', 'val_truth', batch_size=32)
        self.test_ds = lambda: batched_gen('test_blur', 'test_truth', batch_size=32)


    def load_weights(self):
        self.saver.restore(self.sess, self.save_path)

    def save_weights(self):
        self.saver.save(self.sess, self.save_path)

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        total_batch_nb = 0
        best_val_loss = 100000
        val_err = 10000
        nb_epochs_val_not_improved = 0

        for epoch in range(self.params['epoch']):
            print('\nepoch: {}\n'.format(epoch))
            epoch_batch_nb = 0
            progbar = tf.keras.utils.Progbar(self.params['total_images'])

            for X_low_res, Y_ori in self.train_ds():
                feed_dict = {self.input_x_ph: X_low_res,
                             self.input_y_ph: Y_ori,
                             self.phase_train_ph: True}

                self.optimizer.run(session=self.sess, feed_dict=feed_dict)
                prog_vals = []

                # -----------------------
                # TRAIN ERROR CHECK
                # -----------------------
                if epoch_batch_nb % self.params['check_train_every'] == 0:
                    train_err = self.loss.eval(session=self.sess, feed_dict=feed_dict)
                    prog_vals.append(('train_err', train_err))

                # -----------------------
                # VAL ERROR CHECK
                # -----------------------
                if (epoch_batch_nb + 1) % self.params['check_val_every'] == 0:
                    # compute val loss
                    val_err  = self.calculate_val_loss()
                    prog_vals.append(('val_err', val_err))

                    # check early stopped conditions if requested
                    if self.params['early_stopping']:
                        should_save_model, should_stop, best_val_loss, nb_epochs_val_not_improved = self.check_early_stop(val_err, best_val_loss, nb_epochs_val_not_improved)

                        if should_save_model:
                            self.save_weights()

                        if should_stop:
                            print('Early stopping!')
                            return

                # end of batch bookeeping
                progbar.add(len(X_low_res), values=prog_vals if len(prog_vals) > 0 else None)
                epoch_batch_nb += 1
                total_batch_nb += 1

            # -----------------------
            # END OF EPOCH BOOKEEPING
            # -----------------------

            if not self.params['early_stopping']:
                self.save_weights()

        print('done!')


    def calculate_val_loss(self):
        val_losses = []
        for x_val, y_val in self.val_ds():
            feed_dict = {
                self.input_x_ph: x_val,
                self.input_y_ph: y_val,
                self.phase_train_ph: False
            }
            val_losses.append(self.loss.eval(session=self.sess, feed_dict=feed_dict))

        # TODO: normalize val outputs before reporting loss
        return np.mean(val_losses)

    def check_early_stop(self, val_err, best_val_loss, nb_epochs_val_not_improved):
        new_best_val_loss = best_val_loss
        new_nb_epochs_val_not_improved = nb_epochs_val_not_improved + 1
        should_save_model = False

        val_improved = val_err < self.params['early_stop_ratio'] * best_val_loss
        if val_improved:
            new_nb_epochs_val_not_improved = 0
            new_best_val_loss = val_err
            should_save_model = True

        should_stop = new_nb_epochs_val_not_improved > self.params['stale_after']
        return should_save_model, should_stop, new_best_val_loss, new_nb_epochs_val_not_improved

    def eval(self):
        pass

    def predict(self, load_weights=False):
        if load_weights:
            self.load_weights()

        preds = []
        for x_test, y_test in self.test_ds():
            feed_dict = {self.input_x_ph: x_test, self.phase_train_ph: False}
            y_hat_test = self.sess.run(self.model, feed_dict=feed_dict)
            preds.append(y_hat_test)

        # TODO: normalize preds to [0, 1] before returning
        return np.concatenate(preds, axis=0)


    def close():
        self.sess.close()
        if self.h5fp is not None:
            self.h5fp.close()


    def _build_cae_nn(self):
        drop_rate = self.params['dropout_rate']
        encoder = self._encoder_nn(nn_input=self.input_x_ph, drop_rate=drop_rate, phase_train=self.phase_train_ph)
        cae = self._decoder_nn(encoder, drop_rate=drop_rate, phase_train=self.phase_train_ph)
        self.model = cae


    def _encoder_nn(self, nn_input, drop_rate, phase_train):
        with tf.variable_scope('encoder_nn') as scope:
            # layer 1
            enc_1 = tf.layers.conv2d(inputs=nn_input, filters=64, kernel_size=7, strides=[2, 2], padding='same')
            enc_1 = tf.layers.batch_normalization(inputs=enc_1, training=phase_train)
            enc_1 = tf.nn.relu(features=enc_1)

            # layer 2
            enc_2 = tf.layers.conv2d(inputs=enc_1, filters=128, kernel_size=5, strides=[2, 2], padding='same')
            enc_2 = tf.layers.batch_normalization(inputs=enc_2, training=phase_train)
            enc_2 = tf.nn.relu(features=enc_2)
            enc_2 = tf.layers.dropout(inputs=enc_2, rate=drop_rate, training=phase_train)

            # layer 3
            enc_3 = tf.layers.conv2d(inputs=enc_2, filters=256, kernel_size=3, strides=[2, 2], padding='same')
            enc_3 = tf.layers.batch_normalization(inputs=enc_3, training=phase_train)
            enc_3 = tf.nn.relu(features=enc_3)
            enc_3 = tf.layers.dropout(inputs=enc_3, rate=drop_rate, training=phase_train)

            # layer 4
            enc_4 = tf.layers.conv2d(inputs=enc_3, filters=256, kernel_size=3, strides=[2, 2], padding='same')
            enc_4 = tf.layers.batch_normalization(inputs=enc_4, training=phase_train)
            enc_4 = tf.nn.relu(features=enc_4)
            enc_4 = tf.layers.dropout(inputs=enc_4, rate=drop_rate, training=phase_train, name='encoder_out')

        return enc_4

    def _decoder_nn(self, enc_n, drop_rate, phase_train):
        with tf.variable_scope('decoder_nn') as scope:

            # layer 1
            dec_1 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(enc_n)
            dec_1 = tf.layers.conv2d(inputs=dec_1, filters=256, kernel_size=3, strides=[1, 1], padding='same')
            dec_1 = tf.layers.batch_normalization(inputs=dec_1, training=phase_train)
            dec_1 = tf.nn.relu(features=dec_1)
            dec_1 = tf.layers.dropout(inputs=dec_1, training=phase_train, rate=drop_rate)

            # layer 2
            dec_2 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(dec_1)
            dec_2 = tf.layers.conv2d(inputs=dec_2, filters=128, kernel_size=3, strides=[1, 1], padding='same')
            dec_2 = tf.layers.batch_normalization(inputs=dec_2, training=phase_train)
            dec_2 = tf.nn.relu(features=dec_2)
            dec_2 = tf.layers.dropout(inputs=dec_2, training=phase_train, rate=drop_rate)

            # layer 3
            dec_3 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(dec_2)
            dec_3 = tf.layers.conv2d(inputs=dec_3, filters=64, kernel_size=5, strides=[1, 1], padding='same')
            dec_3 = tf.layers.batch_normalization(inputs=dec_3, training=phase_train)
            dec_3 = tf.nn.relu(features=dec_3)
            dec_3 = tf.layers.dropout(inputs=dec_3, training=phase_train, rate=drop_rate)

            # layer 4
            dec_4 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format='channels_last')(dec_3)
            dec_4 = tf.layers.conv2d(inputs=dec_4, filters=1, kernel_size=7, strides=[1, 1], padding='same')

            # out layer
            dec_4 = tf.layers.batch_normalization(inputs=dec_4, name='cae_out', training=phase_train)
            dec_4 = tf.keras.layers.Cropping2D(((1,0), (7, 7)))(dec_4)
            tf.add_to_collection('predict_opt', dec_4)

        return dec_4


class SRGAN:
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
        self.save_path = 'save/srgan_model/model.ckpt'
        self.h5py = None
        # self.log_path = 'save/srgan_model/log/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        input_shape = (None, 95, 146, 1)
        self.input_x_ph = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_x')
        self.input_y_ph = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input_y')
        self.phase_train_ph = tf.placeholder(tf.bool, name='phase_train_ph')

        self._build_srgan_nn()
        self.vgg = tf.keras.applications.VGG19(include_top=False, input_shape=(95, 146, 3))
        self.epsilon = 1e-12

        self.gen_loss = self._generator_loss(self.model['generator'], self.input_y_ph, self.model['discriminator_fake'])
        self.disc_loss = self._discriminator_loss(self.model['discriminator_fake'], self.model['discriminator_real'])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=3)



    def load_data(self, path):
        self.h5fp = h5py.File(path, 'r')

        def batched_gen(blur_name, truth_name, batch_size, chunk_factor = 10, shuffle=False):
            fp = self.h5fp
            data_len = len(fp[blur_name])
            chunk_size = batch_size * chunk_factor
            total_chunks = math.ceil(data_len / chunk_size)

            for chunk_idx in range(total_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, data_len)

                blur_chunk = fp[blur_name][start:end]
                truth_chunk = fp[truth_name][start:end]

                if shuffle:
                    rand_idx, _ = sample_idxs(end - start, 0)
                    blur_chunk = blur_chunk[rand_idx]
                    truth_chunk = truth_chunk[rand_idx]

                chunk_len = len(blur_chunk)
                total_batches = math.ceil(chunk_len / batch_size)
                for batch_idx in range(total_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, chunk_len)

                    blur_batch = blur_chunk[start:end]
                    truth_batch = truth_chunk[start:end]
                    yield blur_batch, truth_batch

        self.train_ds = lambda: batched_gen('train_blur', 'train_truth', batch_size=32, shuffle=True)
        self.val_ds = lambda: batched_gen('val_blur', 'val_truth', batch_size=32)
        self.test_ds = lambda: batched_gen('test_blur', 'test_truth', batch_size=32)

    def get_vgg_feat(self, input, layer_name='block5_conv4'):
        input_3chan = tf.concat((input, input, input), axis=-1)
        out_layer = [layer.output for layer in self.vgg.layers if layer.name == layer_name]
        extract_model = tf.keras.Model(self.vgg.input, out_layer)

        feat = extract_model(input_3chan)
        return feat

    def load_weights(self):
        self.saver.restore(self.sess, self.save_path)

    def save_weights(self):
        self.saver.save(self.sess, self.save_path)

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        total_batch_nb = 0
        best_val_loss = 100000
        val_err = 10000
        nb_epochs_val_not_improved = 0

        for epoch in range(self.params['epoch']):
            print('\nepoch: {}\n'.format(epoch))
            epoch_batch_nb = 0
            progbar = tf.keras.utils.Progbar(self.params['total_images'])

            for X_low_res, Y_ori in self.train_ds():
                feed_dict = {self.input_x_ph: X_low_res,
                             self.input_y_ph: Y_ori,
                             self.phase_train_ph: True}

                self.optimizer.minimize(self.gen_loss).run(session=self.sess, feed_dict=feed_dict)
                self.optimizer.minimize(self.disc_loss).run(session=self.sess, feed_dict=feed_dict)
                prog_vals = []

                # -----------------------
                # TRAIN ERROR CHECK
                # -----------------------
                if epoch_batch_nb % self.params['check_train_every'] == 0:
                    gen_train_err = self.gen_loss.eval(session=self.sess, feed_dict=feed_dict)
                    prog_vals.append(('gen_train_err', train_err))

                    disc_train_err = self.disc_loss.eval(session=self.sess, feed_dict=feed_dict)
                    prog_vals.append(('disc_train_err', train_err))

                # -----------------------
                # VAL ERROR CHECK
                # -----------------------
                if (epoch_batch_nb + 1) % self.params['check_val_every'] == 0:
                    # compute val loss
                    val_err  = self.calculate_val_loss()
                    prog_vals.append(('val_err', val_err))

                    # check early stopped conditions if requested
                    if self.params['early_stopping']:
                        should_save_model, should_stop, best_val_loss, nb_epochs_val_not_improved = self.check_early_stop(val_err, best_val_loss, nb_epochs_val_not_improved)

                        if should_save_model:
                            self.save_weights()

                        if should_stop:
                            print('Early stopping!')
                            return

                # end of batch bookeeping
                progbar.add(len(X_low_res), values=prog_vals if len(prog_vals) > 0 else None)
                epoch_batch_nb += 1
                total_batch_nb += 1

            # -----------------------
            # END OF EPOCH BOOKEEPING
            # -----------------------

            if not self.params['early_stopping']:
                self.save_weights()

        print('done!')

    def _generator_loss(self, gen_fake, target, discrim_fake):
        diff = self.get_vgg_feat(gen_fake) - self.get_vgg_feat(target)
        content_loss = 0.0061 * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=3))
        advers_loss = tf.reduce_mean(-tf.log(discrim_fake + self.epsilon))

        total_loss =  content_loss + 0.001 * advers_loss
        return total_loss

    def _discriminator_loss(self, discrim_fake, discrim_real):
        fake_loss = tf.log(1 - discrim_fake + self.epsilon)
        real_loss = tf.log(discrim_real + self.epsilon)

        total_loss = tf.reduce_mean(- (fake_loss + real_loss))
        return total_loss


    def calculate_val_loss(self):
        val_losses = []
        for x_val, y_val in self.val_ds():
            feed_dict = {
                self.input_x_ph: x_val,
                self.input_y_ph: y_val,
                self.phase_train_ph: False
            }
            val_losses.append(self.gen_loss.eval(session=self.sess, feed_dict=feed_dict))

        # TODO: normalize val outputs before reporting loss?
        return np.mean(val_losses)

    def check_early_stop(self, val_err, best_val_loss, nb_epochs_val_not_improved):
        new_best_val_loss = best_val_loss
        new_nb_epochs_val_not_improved = nb_epochs_val_not_improved + 1
        should_save_model = False

        val_improved = val_err < self.params['early_stop_ratio'] * best_val_loss
        if val_improved:
            new_nb_epochs_val_not_improved = 0
            new_best_val_loss = val_err
            should_save_model = True

        should_stop = new_nb_epochs_val_not_improved > self.params['stale_after']
        return should_save_model, should_stop, new_best_val_loss, new_nb_epochs_val_not_improved

    def eval(self):
        pass

    def predict(self, load_weights=False):
        if load_weights:
            self.load_weights()

        preds = []
        for x_test, y_test in self.test_ds():
            feed_dict = {self.input_x_ph: x_test, self.phase_train_ph: False}
            y_hat_test = self.sess.run(self.model['generator'], feed_dict=feed_dict)
            preds.append(y_hat_test)

        # TODO: normalize preds to [0, 1] before returning?
        return np.concatenate(preds, axis=0)


    def close():
        self.sess.close()
        if self.h5fp is not None:
            self.h5fp.close()


    def _build_srgan_nn(self):
        self.model = {
            'generator': self._build_generator(input=self.input_x_ph,
                                               phase_train=self.phase_train_ph),
            'discriminator_real': self._build_discriminator(input=self.input_y_ph,
                                                            phase_train=self.phase_train_ph),
        }

        self.model['discriminator_fake'] = self._build_discriminator(input=self.model['generator'],
                                                                     phase_train=self.phase_train_ph)

    def _build_generator(self, input, phase_train, out_filters=1):
        with tf.variable_scope('generator_unit', reuse=tf.compat.v1.AUTO_REUSE):

            with tf.variable_scope('input_stage'):
                net = tf.layers.conv2d(input, kernel_size=9, filters=64, strides=1, name='conv', padding='same')
                net = tf.keras.layers.PReLU()(net)
                stage1_output = net

        for i in range(16):
            name_scope = 'resblock_%d'%(i)
            net = self._resid_block(net, phase_train, name_scope)

        with tf.variable_scope('resblock_output'):
            net = tf.layers.conv2d(input, kernel_size=3, filters=64, strides=1, use_bias=False, name='conv_1', padding='same')
            net = tf.layers.batch_normalization(inputs=net, training=phase_train)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            net = tf.layers.conv2d(net, kernel_size=3, filters=256, strides=1, name='conv', padding='same')
            net = tf.nn.depth_to_space(net, block_size=2)
            net = tf.keras.layers.PReLU()(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = tf.layers.conv2d(net, kernel_size=3, filters=256, strides=1, name='conv', padding='same')
            net = tf.nn.depth_to_space(net, block_size=2)
            net = tf.keras.layers.PReLU()(net)

        with tf.variable_scope('output_stage'):
            # TODO: may need to play aroud with downsampling op
            net = tf.layers.conv2d(net, kernel_size=3, filters=128, strides=2, name='conv_1', padding='same')
            net = tf.layers.conv2d(net, kernel_size=3, filters=64, strides=2, name='conv_2', padding='same')
            net = tf.layers.conv2d(net, kernel_size=9, filters=out_filters, strides=1, name='conv_3', padding='same')

        return net


    def _resid_block(self, input, phase_train, scope):
        with tf.variable_scope(scope):
            net = tf.layers.conv2d(input, kernel_size=3, filters=64, strides=1, use_bias=False, name='conv_1', padding='same')
            net = tf.layers.batch_normalization(inputs=net, training=phase_train)
            net = tf.keras.layers.PReLU()(net)
            net = tf.layers.conv2d(input, kernel_size=3, filters=64, strides=1, use_bias=False, name='conv_2', padding='same')
            net = tf.layers.batch_normalization(inputs=net, training=phase_train)

            net = net + input
            return net

    def _build_discriminator(self, input, phase_train):
        with tf.variable_scope('discriminator', reuse=tf.compat.v1.AUTO_REUSE):

            with tf.variable_scope('input_stage'):
                net = tf.layers.conv2d(input, kernel_size=3, filters=64, strides=1, name='conv', padding='same')
                net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)

            net = self._disc_block(net, phase_train, filters=64, kern_size=3, stride=2, scope='disblock_1')
            net = self._disc_block(net, phase_train, filters=128, kern_size=3, stride=1, scope='disblock_2')
            net = self._disc_block(net, phase_train, filters=128, kern_size=3, stride=2, scope='disblock_3')
            net = self._disc_block(net, phase_train, filters=256, kern_size=3, stride=1, scope='disblock_4')
            net = self._disc_block(net, phase_train, filters=256, kern_size=3, stride=2, scope='disblock_5')
            net = self._disc_block(net, phase_train, filters=512, kern_size=3, stride=1, scope='disblock_6')
            net = self._disc_block(net, phase_train, filters=512, kern_size=3, stride=2, scope='disblock_7')

            with tf.variable_scope('dense_layer_1'):
                net = tf.layers.flatten(net)
                net = tf.layers.dense(net, 1024, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)

            with tf.variable_scope('dense_layer_2'):
                net = tf.layers.dense(net, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
                net = tf.math.sigmoid(net)

        return net


    def _disc_block(self, input, phase_train, filters, kern_size, stride, scope):
        with tf.variable_scope(scope):
            net = tf.layers.conv2d(input, kernel_size=kern_size, filters=filters, strides=stride, use_bias=False, name='conv', padding='same')
            net = tf.layers.batch_normalization(inputs=net, training=phase_train)
            net = tf.keras.layers.LeakyReLU(alpha=0.2)(net)

            return net
