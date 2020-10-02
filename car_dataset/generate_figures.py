
import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import training.misc as misc


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)



def draw_style_mixing_figure_transition(png, Gs, w, h, style1_seeds, style2_seed, style_ranges):
    print(png)
    style1_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in style1_seeds)
    style2_latents = np.stack(np.random.RandomState(style2_seed).randn(Gs.input_shape[1]) for _ in range(len(style_ranges)))
    style1_dlatents = Gs.components.mapping.run(style1_latents, None) # [seed, layer, component]
    style2_dlatents = Gs.components.mapping.run(style2_latents, None)  # [seed, layer, component]
    style1_images = Gs.components.synthesis.run(style1_dlatents, randomize_noise=False, **synthesis_kwargs)
    style2_image = Gs.components.synthesis.run(style2_dlatents, randomize_noise=False, **synthesis_kwargs)[0]

    canvas = PIL.Image.new('RGB', (w * (len(style1_seeds) + 1), h * (len(style_ranges) + 1)), 'white')
    for col, src_image in enumerate(list(style1_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row in range(len(style_ranges)):
        canvas.paste(PIL.Image.fromarray(style2_image, 'RGB'), (0, (row + 1) * h))
        for col in range(len(style1_seeds)):
            mixed_dlatent = np.array([style1_dlatents[col]])
            mixed_dlatent[:, style_ranges[row], :] = style2_dlatents[row, style_ranges[row], :]
            image = Gs.components.synthesis.run(mixed_dlatent, randomize_noise=False, **synthesis_kwargs)[0]
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 4: Noise detail.

def draw_noise_detail_figure(png, Gs, w, h, num_samples, seeds):
    print(png)
    canvas = PIL.Image.new('RGB', (w * 2, h * len(seeds)), 'white')
    for row, seed in enumerate(seeds):
        latents = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1])] * num_samples)
        images = Gs.run(latents, None, truncation_psi=1, **synthesis_kwargs)
        canvas.paste(PIL.Image.fromarray(images[0], 'RGB'), (0, row * h))
        diff = np.std(np.mean(images, axis=3), axis=0) * 4
        diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
        canvas.paste(PIL.Image.fromarray(diff, 'L'), (w, row * h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 5: Noise components.

def draw_noise_components_figure(png, Gs, w, h, seeds, noise_ranges, flips):
    print(png)
    Gsc = Gs.clone()
    noise_vars = [var for name, var in Gsc.components.synthesis.vars.items() if name.startswith('noise')]
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars))) # [(var, val), ...]
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
    all_images = []
    for noise_range in noise_ranges:
        tflib.set_vars({var: val * (1 if i in noise_range else 0) for i, (var, val) in enumerate(noise_pairs)})
        range_images = Gsc.run(latents, None, truncation_psi=1, randomize_noise=False, **synthesis_kwargs)
        range_images[flips, :, :] = range_images[flips, :, ::-1]
        all_images.append(list(range_images))

    canvas = PIL.Image.new('RGB', (w * 2, h * len(seeds)), 'white')
    for col, col_images in enumerate(zip(*all_images)):
        canvas.paste(PIL.Image.fromarray(col_images[0], 'RGB'), (0, col * h))
        canvas.paste(PIL.Image.fromarray(col_images[1], 'RGB'), (w, col * h))
        # canvas.paste(PIL.Image.fromarray(col_images[1], 'RGB'), (col * w + w//2, 0))
        #canvas.paste(PIL.Image.fromarray(col_images[2], 'RGB').crop((0, 0, w//2, h)), (col * w, h))
        #canvas.paste(PIL.Image.fromarray(col_images[3], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 8: Truncation trick.

def draw_truncation_trick_figure(png, Gs, w, h, seeds, psis):
    print(png)
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
    dlatents = Gs.components.mapping.run(latents, None) # [seed, layer, component]
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]

    canvas = PIL.Image.new('RGB', (w * len(psis), h * len(seeds)), 'white')
    for row, dlatent in enumerate(list(dlatents)):
        row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col * w, row * h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Main program.

def main():
    tflib.init_tf()
    baseline_network_pkl = '../results/00002-sgan-car512-2gpu/network-snapshot-023949.pkl'
    no_noise_network_pkl = '../../results/00022-sgan-ffhq256-2gpu-no-noise/network-snapshot-014926.pkl'

    _G, _D, Gs_baseline = misc.load_pkl(baseline_network_pkl)
    # _G, _D, Gs_no_nosie = misc.load_pkl(no_noise_network_pkl)

    # 888,1733
    # draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure02-uncurated-ffhq.png'), Gs, cx=0, cy=0, cw=256, ch=256, rows=3, lods=[0,1,2,2,3,3], seed=5)
    #draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), Gs, w=256, h=256, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(0,4)]*3+[range(4,8)]*2+[range(8,18)])
    # draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), Gs, w=256, h=256, src_seeds=[123,456,789], dst_seeds=[888,1733]*2, style_ranges=[range(0,4)]*2+[range(4,8)]*2)


    # draw_style_mixing_figure_transition(os.path.join(config.result_dir, 'no-style-mixing.png'), Gs_baseline, w=256, h=256, style1_seeds=[222, 1733, 4], style2_seed=[888], style_ranges=[list(range(i-2, i)) for i in range(2, 16, 2)])
    #draw_style_mixing_figure_transition(os.path.join(config.result_dir, 'no-style-mixing.png'), Gs_no_style_mix, w=256, h=256, style1_seeds=[12, 23, 34], style2_seed=[45], style_ranges=[list(range(i-2, i)) for i in range(2, 16, 2)])

    draw_noise_detail_figure('images/noise_detail.png', Gs_baseline, w=512, h=512, num_samples=100, seeds=[5, 4])
    draw_noise_components_figure('images/noise-components.png', Gs_baseline, w=512, h=512, seeds=[5, 4], noise_ranges=[range(0, 14), range(0, 0)], flips=[])
    #draw_truncation_trick_figure(os.path.join(config.result_dir, 'figure08-truncation-trick.png'), Gs, w=256, h=256, seeds=[92,388], psis=[1, 0.7, 0.5, 0, -0.5, -1])


    #draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure10-uncurated-bedrooms.png'), Gs, cx=0, cy=0, cw=256, ch=256, rows=5, lods=[0,0,1,1,2,2,2], seed=0)
    #draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure11-uncurated-cars.png'), Gs, cx=0, cy=64, cw=512, ch=384, rows=4, lods=[0,1,2,2,3,3], seed=2)
    #draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure12-uncurated-cats.png'), Gs, cx=0, cy=0, cw=256, ch=256, rows=5, lods=[0,0,1,1,2,2,2], seed=1)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
