"""
Script to decode images

authors:
    William Tong (wlt2115@columbia.edu)
    Young Joon Kim (yk2611@columbia.edu)
date: March 5, 2020
"""

# <codecell>
from internal.joon import *

# <codecell>
spike_train_file = "/home/grandpaa/workspace/neural_decoding/dataset/2017_11_29/yass_spiketrain.npy"
image_times_file = "/home/grandpaa/workspace/neural_decoding/dataset/2017_11_29/trigger_times_201711290.mat"
sorted_units_file = ""
save_dir = "save/"
drop_no = 0

make_neural_matrix(spike_train_file, image_times_file, sorted_units_file, drop_no, save_dir)

# <codecell>
movie_file = "/home/grandpaa/workspace/neural_decoding/dataset/2017_11_29/ImageNet_stix2_0_045.mat"
save_dir = "save/"

make_image_matrix(movie_file, save_dir)
make_smooth_image_matrix(movie_file, save_dir)

# <codecell>
neural_matrix_file = "save/train_neural_matrix.npy"
image_matrix_file = "save/smooth_train_images.npy"
save_dir = "save/"

fraction = 0
radius = 0

make_weights(neural_matrix_file, image_matrix_file, 0, 7700, save_dir)

# <codecell>
neural_matrix_file = "save/test_neural_matrix.npy"
weights_file = "save/smooth_weights.npy"
save_dir = "save/"
index=0

decode_images(neural_matrix_file, weights_file, index, save_dir)










# <codecell>
