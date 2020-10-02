from training import misc
import numpy as np
import dnnlib.tflib as tflib
from matplotlib import pyplot as plt
import cv2


G_args = {}
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

tflib.init_tf()
# baseline model
baseline_network_pkl = '../../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl'


_G, _D, Gs = misc.load_pkl(baseline_network_pkl)
G_baseline = tflib.Network('G', func_name='training.networks_stylegan_cutoff.G_style', num_channels=3, resolution=256,
                  label_size=0, structure='linear', **G_args)
G_baseline.copy_vars_from(Gs)


# no noise model
without_noise_network_pkl = '../../results/00022-sgan-ffhq256-2gpu/network-snapshot-014926.pkl'

_G, _D, Gs = misc.load_pkl(baseline_network_pkl)
G_without_noise = tflib.Network('G', func_name='training.networks_stylegan_cutoff.G_style', num_channels=3, resolution=256,
                  label_size=0, structure='linear', **G_args)
G_without_noise.copy_vars_from(Gs)

num_images = 1000
latents = np.stack(np.random.randn(Gs.input_shape[1]) for _ in range(num_images))
images_baseline = G_baseline.run(latents, None, **synthesis_kwargs)
images_without_noise = G_without_noise.run(latents, None, **synthesis_kwargs)


normal_edges = cv2.Sobel(images_baseline[0, :, :, 0], cv2.CV_64F, 1, 1, ksize=3)
normal_edges += np.abs(np.min(normal_edges))
normal_edges /= np.max(normal_edges)
normal_edges *= 255
plt.imshow(normal_edges)
plt.axis('off')
plt.show()

magnitude_baseline = 0
magnitude_without_noise = 0
num_filters = images_baseline.shape[-1]

for j in range(num_images):
    for i in range(num_filters):
        sobel_baseline = cv2.Sobel(images_baseline[j, :, :, i], cv2.CV_64F, 1, 1, ksize=3)
        sobel_baseline = np.abs(sobel_baseline)
        # sobel_baseline = np.clip(sobel_baseline, 0, 255)
        # sobel_baseline += np.abs(np.min(sobel_baseline))
        magnitude_baseline += np.sum(sobel_baseline) / (num_filters * 256 * 256)

        sobel_without_noise = cv2.Sobel(images_without_noise[j, :, :, i], cv2.CV_64F, 1, 1, ksize=3)#
        sobel_without_noise = np.abs(sobel_without_noise)
        # sobel_without_noise = np.clip(sobel_without_noise, 0, 255)
        # sobel_without_noise += np.abs(np.min(sobel_without_noise))
        magnitude_without_noise += np.sum(sobel_without_noise) / (num_filters * 256 * 256)

print('sobel magn')
print(magnitude_baseline / num_images)
print(magnitude_without_noise / num_images)

print()
print('mean')
print(np.mean(images_baseline))
print(np.mean(images_without_noise))

std_baseline = 0
std_without_noise = 0

for j in range(num_images):
    for i in range(num_filters):
        std_baseline += np.std(images_baseline[j, :, :, i])
        std_without_noise += np.std(images_without_noise[j, :, :, i])

print()
print('std')
print(std_baseline / (num_images + num_filters))
print(std_without_noise / (num_images + num_filters))
