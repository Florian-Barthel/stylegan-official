from training import misc
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
from matplotlib import pyplot as plt


run_id = 15
snapshot = 15326
G_args = {}
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

tflib.init_tf()
# baseline model
# network_pkl = '../../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl'
network_pkl = '../../results/00046-sgan-ffhq256-2gpu-adain-pixel-norm-continue/network-snapshot-012126.pkl'

# no noise model
# network_pkl = 'results/00022-sgan-ffhq256-2gpu/network-snapshot-005726.pkl'
_G, _D, Gs = misc.load_pkl(network_pkl)
G = tflib.Network('G', func_name='training.networks_stylegan_cutoff.G_style', num_channels=3, resolution=256,
                  label_size=0, structure='linear', **G_args)
G.copy_vars_from(Gs)

G_original = tflib.Network('G', func_name='training.networks_stylegan.G_style', num_channels=3, resolution=256,
                  label_size=0, structure='linear', **G_args)
G_original.copy_vars_from(Gs)

latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in [8])
images = G.run(latents, None, use_instance_norm = False, **synthesis_kwargs)
images_original = G_original.run(latents, None, use_instance_norm = False, **synthesis_kwargs)
print(images.shape)
fig, axs = plt.subplots(3, 3)
im = images[0]

counter = 20
for i in range(3):
    for j in range(3):
        axs[i][j].imshow(im[:, :, counter])
        axs[i][j].axis('off')
        counter += 1

plt.savefig('water_droplet_removed_adain_pixel.png', dpi=300)
plt.show()

plt.imshow(images_original[0])
plt.axis('off')
plt.savefig('water_droplet_output_removed_adain_pixel.png', dpi=300)
plt.show()
plt.close('all')
print(np.min(im))
print(np.max(im))
print(np.mean(im))
