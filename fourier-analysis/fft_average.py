import training.misc as misc
import numpy as np
import dnnlib.tflib as tflib
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import cv2
from matplotlib import cm

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)


def single_image(Gs, num_images):
    latents = np.random.normal(0.0, 1.0, [num_images, Gs.input_shape[1]])
    # change to use_noise = False when executing the new model
    return Gs.run(latents, None, use_noise=False, **synthesis_kwargs)

# real_img = cv2.imread('E:/ffhq_256/00005.png', 0)

tflib.init_tf()
no_noise_network_pkl = '../results/00022-sgan-ffhq256-2gpu-no-noise/network-snapshot-014926.pkl'
baseline_network_pkl = '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl'
_G, _D, Gs = misc.load_pkl(no_noise_network_pkl)



def get_fft(bild):
    bildf64 = bild.astype('float64')
    fft = np.fft.fft2(bildf64)
    return np.abs(fft)


def get_dct(bild):
    bildf64 = bild.astype('float64')
    return np.abs(fftpack.dct(bildf64))


mean_img = np.zeros([256, 256], dtype=np.float64)
num_images = 1000
for i in range(num_images):
    fake_img = single_image(Gs, num_images=1)[0]
    fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
    bildf64 = fake_img.astype('float64')
    fft = np.fft.fft2(bildf64)
    fftmag = np.abs(fft)
    mean_img += np.fft.fftshift(np.log(fftmag + 0.00001))
    # bildf64 = fake_img.astype('float64')
    # mean_img += np.log(np.abs(fftpack.dct(bildf64)) + 0.00001)

mean_img /= num_images

print(np.max(mean_img))
print(np.min(mean_img))
mean_img = np.clip(mean_img, 3.0, 9.0)
plt.imshow(mean_img)
plt.savefig('fourier_no_noise_2.png')
plt.show()
