import training.misc as misc
import numpy as np
import dnnlib.tflib as tflib
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm

G_args = {}
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)


def single_image(Gs, num_images):
    latents = np.random.normal(0.0, 1.0, [num_images, Gs.input_shape[1]])
    # change to use_noise = False when executing the new model
    return Gs.run(latents, None, use_noise=False, **synthesis_kwargs)

# real_img = cv2.imread('E:/ffhq_256/00005.png', 0)

tflib.init_tf()
baseline_network_pkl = '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl'

no_progan_network_pkl = '../results/00001-sgan-ffhq256-2gpu-remove-progan/network-snapshot-014800.pkl'

_G, _D, Gs = misc.load_pkl(baseline_network_pkl)
G_cutoff = tflib.Network('G', func_name='training.networks_stylegan_cutoff.G_style', num_channels=3, resolution=256,
                  label_size=0, structure='linear', **G_args)
G_cutoff.copy_vars_from(Gs)

mean_img = np.zeros([256, 256], dtype=np.float64)

num_images = 1000
latents = np.stack(np.random.randn(Gs.input_shape[1]) for _ in range(num_images))
images = G_cutoff.run(latents, None, **synthesis_kwargs)

num_filters = images.shape[-1]

for i in range(num_images):
    for j in range(num_filters):
        fake_img = images[i, :, :, j]
        bildf64 = fake_img.astype('float64')
        fft = np.fft.fft2(bildf64)
        fftmag = np.abs(fft)
        mean_img += np.log(fftmag + 0.00001)
        # bildf64 = fake_img.astype('float64')
        # mean_img += np.log(np.abs(fftpack.dct(bildf64)) + 0.00001)

mean_img /= num_images

print(np.max(mean_img))
print(np.min(mean_img))
mean_img = mean_img[:128, :128]
plt.imshow(mean_img)
plt.savefig('fourier_baseline_256.png')
plt.show()

median_index = np.argsort(mean_img)[63][63]
print(median_index)