"""
Code from Joon's Jupyter notebook

authors:
    William Tong (wlt2115@columbia.edu)
    Young Joon Kim (yk2611@columbia.edu)
date: March 5, 2020
"""

import os
import os.path
import h5py
import numpy as np
import scipy.ndimage

from tqdm import tqdm

import matplotlib.pyplot as plt
import math
import sys
import time
import operator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

tnrange = lambda x: tqdm(range(x))

# TODO: tidy and modularize
def make_neural_matrix(spike_train_file, image_times_file, sorted_units_file, drop_no, save_dir):
    train_times = np.asarray(h5py.File(image_times_file, 'r')["image_times"])[0,:]
    train_images = train_times.shape[0]
    train_times = np.append(train_times, 9999)


    #sorted_units = np.load(sorted_units_file)
    #drop_units = sorted_units[-drop_no:]
    #keep_units = sorted_units[:-drop_no]

    spike_train = np.load(spike_train_file)

    ##

    spike_times = spike_train[:,0] / 20000
    units = np.max(spike_train[:,1]).astype(np.int) + 1
    ##

    #### DROP LOWEST WEIGHT XXXX UNITS ####
    #spike_train = spike_train[np.in1d(spike_train[:,1], drop_units, invert = True),:]
    #spike_times = spike_train[:,0] / 20000

    #units = keep_units.shape[0]
    ########


    train_matrix = np.empty((train_images, units * 2))
    bin_edges = np.copy(train_times)

    for i in tnrange(int(train_images)):

        bin1_start = train_times[i] + 0.03
        bin1_end = train_times[i] + 0.15

        bin2_start = train_times[i] + 0.17
        bin2_end = train_times[i] + 0.3

        bin_edges = np.append(bin_edges, bin1_start)
        bin_edges = np.append(bin_edges, bin1_end)
        bin_edges = np.append(bin_edges, bin2_start)
        bin_edges = np.append(bin_edges, bin2_end)

    bin_edges = np.sort(bin_edges)
    print(bin_edges.shape)

    hist_spikes, hist_edges = np.histogram(spike_times, bin_edges)

    count = spike_times.shape[0] - np.sum(hist_spikes)

    for i in tnrange(hist_spikes.shape[0]):

        bin_count = hist_spikes[i]

        if i%5 == 0:
            count += bin_count

        elif i%5 == 1:
            for j in range(bin_count):

                unit = int(spike_train[count,1])
                train_matrix[i//5,unit*2] += 1
                count +=1

                ###### DROP UNTIS ###
                #unit = int(spike_train[count,1])
                #index = np.where(keep_units == unit)[0]
                #train_matrix[i//5, index*2] += 1
                #count += 1

                ####

        elif i%5 == 2:
            count += bin_count

        elif i%5==3:
            for j in range(bin_count):

                unit = int(spike_train[count,1])
                train_matrix[i//5,unit*2+1] += 1
                count +=1

                ###### DROP UNTIS ###
                #unit = int(spike_train[count,1])
                #index = np.where(keep_units == unit)[0]
                #train_matrix[i//5, index*2+1] += 1
                #count += 1

                ####

        elif i%5==4:
            count += bin_count

    print(count)
    print(spike_train.shape)

    train_matrix = np.concatenate((train_matrix, np.ones((train_images, 1))), axis=1)

    test_matrix = train_matrix[9900:,:]
    valid_matrix = train_matrix[9800:9900, :]
    train_matrix = train_matrix[:9800,:]

    print(train_matrix.shape)
    print(test_matrix.shape)
    print(valid_matrix.shape)
    print(np.sum(train_matrix) + np.sum(test_matrix) + np.sum(valid_matrix))

    np.save(os.path.join(save_dir, "train_neural_matrix.npy"), train_matrix)
    np.save(os.path.join(save_dir, "test_neural_matrix.npy"), test_matrix)
    np.save(os.path.join(save_dir, "valid_neural_matrix.npy"), valid_matrix)


def make_resid_neural_matrix(spike_train_file, resid_file, image_times_file, save_dir):

    train_times = np.asarray(h5py.File(image_times_file, 'r')["image_times"])[0,:]
    train_images = train_times.shape[0]
    train_times = np.append(train_times, 9999)

    spike_train = np.load(spike_train_file)
    residuals = np.load(resid_file)
    found_units = int(np.max(spike_train[:,1])+1)
    residuals[:, 1] += found_units

    full_spike_train = np.vstack((spike_train, residuals))
    full_spike_train = full_spike_train[full_spike_train[:,0].argsort()]

    full_spike_times = full_spike_train[:,0] / 20000
    full_units = np.max(full_spike_train[:,1]).astype(np.int) + 1

    train_matrix = np.empty((train_images, full_units * 2))
    bin_edges = np.copy(train_times)

    for i in tnrange(int(train_images)):

        bin1_start = train_times[i] + 0.03
        bin1_end = train_times[i] + 0.15

        bin2_start = train_times[i] + 0.17
        bin2_end = train_times[i] + 0.3

        bin_edges = np.append(bin_edges, bin1_start)
        bin_edges = np.append(bin_edges, bin1_end)
        bin_edges = np.append(bin_edges, bin2_start)
        bin_edges = np.append(bin_edges, bin2_end)

    bin_edges = np.sort(bin_edges)
    print(bin_edges.shape)

    hist_spikes, hist_edges = np.histogram(full_spike_times, bin_edges)

    count = full_spike_times.shape[0] - np.sum(hist_spikes)

    for i in tnrange(hist_spikes.shape[0]):

        bin_count = hist_spikes[i]

        if i%5 == 0:
            count += bin_count

        elif i%5 == 1:
            for j in range(bin_count):

                unit = int(full_spike_train[count,1])
                train_matrix[i//5,unit*2] += 1
                count +=1

        elif i%5 == 2:
            count += bin_count

        elif i%5==3:
            for j in range(bin_count):

                unit = int(full_spike_train[count,1])
                train_matrix[i//5,unit*2+1] += 1
                count +=1

        elif i%5==4:
            count += bin_count

    print(count)
    print(full_spike_train.shape)

    train_matrix = np.concatenate((train_matrix, np.ones((train_images, 1))), axis=1)

    test_matrix = train_matrix[9900:,:]
    valid_matrix = train_matrix[9800:9900, :]
    train_matrix = train_matrix[:9800,:]

    print(train_matrix.shape)
    print(test_matrix.shape)
    print(valid_matrix.shape)
    print(np.sum(train_matrix) + np.sum(test_matrix) + np.sum(valid_matrix))

    np.save(os.path.join(save_dir, "resid_ks2_train_neural.npy"), train_matrix)
    np.save(os.path.join(save_dir, "resid_ks2_test_neural.npy"), test_matrix)
    np.save(os.path.join(save_dir, "resid_ks2_valid_neural.npy"), valid_matrix)


def make_image_matrix(movie_file, save_dir):
    movie = np.asarray(h5py.File(movie_file,'r')['movie'])[:,:,32:-32]
    image_no = int(movie.shape[0])

    stim_size0 = movie.shape[1]
    stim_size1 = movie.shape[2]

    image_matrix = np.empty((image_no,95 * 146)) ##### 95 x 146 #$###

    for i in tnrange(image_no):
        image = movie[i,:,:][25:-40,50:-60] ####
        image_matrix[i] = image.flatten()

    train_images = image_matrix[:9800,:]
    test_images = image_matrix[9900:,:]
    valid_images = image_matrix[9800:9900,:]

    print(train_images.shape)
    print(test_images.shape)
    print(valid_images.shape)

    np.save(os.path.join(save_dir,"train_images.npy"), train_images)
    np.save(os.path.join(save_dir,"test_images.npy"), test_images)
    np.save(os.path.join(save_dir,"valid_images.npy"), valid_images)


def make_smooth_image_matrix(movie_file, save_dir):
    movie = np.asarray(h5py.File(movie_file,'r')['movie'])[:,:,32:-32]
    image_no = int(movie.shape[0])

    stim_size0 = movie.shape[1]
    stim_size1 = movie.shape[2]

    image_matrix = np.empty((image_no,95 * 146))

    for i in tnrange(image_no):
        image = movie[i,:,:]
        smooth_image = scipy.ndimage.gaussian_filter(image,
                                                     sigma=4,
                                                     order=0,
                                                     mode='constant',
                                                     cval=0.0,
                                                     truncate=3.0)

        image_matrix[i] = smooth_image[25:-40,50:-60].flatten()

    train_images = image_matrix[:9800,:]
    test_images = image_matrix[9900:,:]
    valid_images = image_matrix[9800:9900,:]

    print(train_images.shape)
    print(test_images.shape)
    print(valid_images.shape)

    np.save(os.path.join(save_dir,"smooth_train_images.npy"), train_images)
    np.save(os.path.join(save_dir,"smooth_test_images.npy"), test_images)
    np.save(os.path.join(save_dir,"smooth_valid_images.npy"), valid_images)


def make_weights(neural_matrix_file, image_matrix_file, index, regularizer, save_dir):
    X = np.load(neural_matrix_file)
    Y = np.load(image_matrix_file)

    XTX = np.matmul(X.T,X)
    XTX_inv = np.linalg.inv(XTX + regularizer * np.identity(XTX.shape[0]))
    XTY = np.matmul(X.T,Y)
    theta = np.matmul(XTX_inv,XTY)

    print(theta.shape)
    np.save(os.path.join(save_dir,"smooth_weights.npy"), theta)

def make_circle_weights(neural_matrix_file, image_matrix_file, index, radius, regularizer, save_dir):
    X = np.load(neural_matrix_file)
    Y = np.load(image_matrix_file)
    sorted_units = np.load(sorted_unit_file)

    XTX = np.matmul(X.T,X)
    XTX_inv = np.linalg.inv(XTX + regularizer * np.identity(XTX.shape[0]))
    XTY = np.matmul(X.T,Y)
    theta = np.matmul(XTX_inv,XTY)

    theta_circle = np.empty((theta.shape[0], theta.shape[1]))
    print(theta.shape)

    for i in range(theta.shape[0]):
        weight = theta[i,:].reshape((95,146))
        abs_max = np.max(np.abs(weight))
        weight_list = np.ndarray.tolist(np.abs(weight))

        for j in range(weight.shape[0]):
            weight_row = weight_list[j]
            if abs_max in weight_row:
                center_x = j
                center_y = weight_row.index(abs_max)
        y,x = np.ogrid[-center_x:95-center_x, -center_y:146-center_y]
        zero_mask = x*x + y*y > radius * radius
        weight[zero_mask]=0

        theta_circle[i,:] = weight.flatten()

    np.save(os.path.join(save_dir,"yass_circle_" + str(index) + "_weights.npy"), theta_circle)


def multi_make_weights(neural_matrix_file, image_matrix_file, fraction, radius, save_dir):
    reg_list = np.logspace(3, 5, 10, endpoint=True)
    for i in tnrange(len(reg_list)):
        regularizer = reg_list[i]
        index = i

        make_weights(neural_matrix_file, image_matrix_file, index, regularizer, save_dir)
        #make_circle_weights(neural_matrix_file, image_matrix_file, index, radius, regularizer, save_dir)


def decode_images(neural_matrix_file, weights_file, index, save_dir):
    X = np.load(neural_matrix_file)
    theta = np.load(weights_file)
    decoded = np.matmul(X, theta)
    np.save(os.path.join(save_dir,"smooth_test_decoded.npy"), decoded)


def multi_decode_images(neural_matrix_file, w1_file, w2_file, w3_file, w4_file,
                       w5_file, w6_file, w7_file, w8_file, w9_file, w10_file, save_dir):

    w_files = [w1_file, w2_file, w3_file, w4_file, w5_file, w6_file, w7_file, w8_file, w9_file, w10_file]

    for i in tnrange(len(w_files)):
        w_file = w_files[i]

        decode_images(neural_matrix_file, w_file, i, save_dir)
