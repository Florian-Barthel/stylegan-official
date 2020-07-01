from training import misc
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os
import config
import dnnlib

run_id = 15
snapshot = 15326
G_args = {}
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

img = cv2.imread('kernel_visualization/image_1.png', 0).astype('float32')
img /= 127.5
img -= 1.0
img = np.expand_dims(img, 0)
img = np.expand_dims(img, -1)
tflib.init_tf()
network_pkl = 'results/00015-sgan-ffhq256-1gpu/network-snapshot-015326.pkl'
_G, _D, Gs = misc.load_pkl(network_pkl)
G = tflib.Network('G', func_name='training.networks_stylegan_cutoff.G_style', num_channels=3, resolution=256, label_size=0, structure='fixed', **G_args)
G.copy_vars_from(Gs)
# filters = Gs.get_var('G_synthesis/256x256/Conv1/weight')
# img = np.full((1, 256, 256, 1), 0.5).astype('float32')
# img = np.random.rand(1, 256, 256, 1).astype('float32')

# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (filters - f_min)

#result = tf.nn.conv2d(img,
#                      tf.expand_dims(tf.expand_dims(filters[:, :, 1, 0], axis=-1), axis=-1),
#                      strides=[1, 1, 1, 1],
#                      padding='SAME')

latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in [123, 100])
# images = G.get_output_for(latents, None, is_validation=True, randomize_noise=True, **synthesis_kwargs)
images = G.run(latents, None, **synthesis_kwargs)
print(images)
im = np.squeeze(images.eval()[0], axis=-1)
plt.show()

im = np.squeeze(images.eval()[0], axis=-1)
im += np.abs(np.min(im))
scale = 255 / np.max(im)
im *= scale
plt.matshow(images[0])
print(np.min(im))
print(np.max(im))
print(np.mean(im))
