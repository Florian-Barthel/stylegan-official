
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import training.misc as misc
import tensorflow as tf

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


#----------------------------------------------------------------------------
# Figures 2, 3, 10, 11, 12: Multi-resolution grid of uncurated result images.

def draw_uncurated_result_figure(png, Gs, seed, psi):
    print(png)
    rows = 7
    cols = 7
    latents = np.random.RandomState(seed).randn(rows * cols, Gs.input_shape[1])
    #images = Gs.run(latents, None, **synthesis_kwargs)

    #latents = tf.random_normal([25] + Gs.input_shape[1:])
    # images = Gs.get_output_for(latents, None, is_validation=True, randomize_noise=False, truncation_psi=0.0, truncation_cutoff=14)
    images = Gs.get_output_for(latents, None, is_validation=True, randomize_noise=False, truncation_psi_val=psi, truncation_cutoff_val=8)
    images = tflib.convert_images_to_uint8(images).eval()

    images = np.transpose(images, (0, 2, 3, 1))
    canvas = PIL.Image.new('RGB', (256 * rows, 256 * cols), 'white')
    image_iter = iter(list(images))
    for col in range(cols):
        for row in range(rows):
            image = PIL.Image.fromarray(next(image_iter), 'RGB')
            canvas.paste(image, (row * 256, col * 256))
    canvas.save(png)




def main():
    tflib.init_tf()
    baseline_network_pkl = '../../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl'
    no_style_mix_network_pkl = '../../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-013726.pkl'

    _G, _D, Gs_baseline = misc.load_pkl(baseline_network_pkl)
    _G, _D, Gs_no_style_mix = misc.load_pkl(no_style_mix_network_pkl)

    # 888,1733
    psi = 0.0
    for _ in range(11):
        draw_uncurated_result_figure('truncation_example_' + str(psi) +'.png', Gs_baseline, seed=13, psi=psi)
        psi += 0.1
    #draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), Gs, w=256, h=256, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(0,4)]*3+[range(4,8)]*2+[range(8,18)])
    # draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), Gs, w=256, h=256, src_seeds=[123,456,789], dst_seeds=[888,1733]*2, style_ranges=[range(0,4)]*2+[range(4,8)]*2)


    # draw_style_mixing_figure_transition(os.path.join(config.result_dir, 'no-style-mixing.png'), Gs_baseline, w=256, h=256, style1_seeds=[222, 1733, 4], style2_seed=[888], style_ranges=[list(range(i-2, i)) for i in range(2, 16, 2)])
    # draw_style_mixing_figure_transition(os.path.join(config.result_dir, 'no-style-mixing.png'), Gs_no_style_mix, w=256, h=256, style1_seeds=[12, 23, 34], style2_seed=[45], style_ranges=[list(range(i-2, i)) for i in range(2, 16, 2)])

    #draw_noise_detail_figure(os.path.join(config.result_dir, 'figure04-noise-detail.png'), Gs, w=256, h=256, num_samples=100, seeds=[1157,1012])
    #draw_noise_components_figure(os.path.join(config.result_dir, 'figure05-noise-components.png'), Gs, w=256, h=256, seeds=[1967,1555], noise_ranges=[range(0, 14), range(0, 0), range(8, 14), range(0, 8)], flips=[1])
    #draw_truncation_trick_figure(os.path.join(config.result_dir, 'figure08-truncation-trick.png'), Gs, w=256, h=256, seeds=[92,388], psis=[1, 0.7, 0.5, 0, -0.5, -1])


    #draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure10-uncurated-bedrooms.png'), Gs, cx=0, cy=0, cw=256, ch=256, rows=5, lods=[0,0,1,1,2,2,2], seed=0)
    #draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure11-uncurated-cars.png'), Gs, cx=0, cy=64, cw=512, ch=384, rows=4, lods=[0,1,2,2,3,3], seed=2)
    #draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure12-uncurated-cats.png'), Gs, cx=0, cy=0, cw=256, ch=256, rows=5, lods=[0,0,1,1,2,2,2], seed=1)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
