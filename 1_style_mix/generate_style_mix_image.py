
import os
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import config
import training.misc as misc


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)


# def draw_style_mixing_figure_transition(png, Gs, w, h, style1_seeds, style2_seed, style_ranges):
#     style1_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in style1_seeds)
#     style2_latents = np.stack(np.random.RandomState(style2_seed).randn(Gs.input_shape[1]) for _ in range(len(style_ranges)))
#     style1_dlatents = Gs.components.mapping.run(style1_latents, None) # [seed, layer, component]
#     style2_dlatents = Gs.components.mapping.run(style2_latents, None)  # [seed, layer, component]
#     style1_images = Gs.components.synthesis.run(style1_dlatents, randomize_noise=False, **synthesis_kwargs)
#     style2_image = Gs.components.synthesis.run(style2_dlatents, randomize_noise=False, **synthesis_kwargs)[0]
#
#     canvas = PIL.Image.new('RGB', (w * (len(style1_seeds) + 1), h * (len(style_ranges) + 1)), 'white')
#     for col, src_image in enumerate(list(style1_images)):
#         canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
#     for row in range(len(style_ranges)):
#         canvas.paste(PIL.Image.fromarray(style2_image, 'RGB'), (0, (row + 1) * h))
#         for col in range(len(style1_seeds)):
#             mixed_dlatent = np.array([style1_dlatents[col]])
#             mixed_dlatent[:, style_ranges[row], :] = style2_dlatents[row, style_ranges[row], :]
#             image = Gs.components.synthesis.run(mixed_dlatent, randomize_noise=False, **synthesis_kwargs)[0]
#             canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
#     canvas.save(png)
#     print(png)

def draw_style_mixing_figure_transition(png, Gs, w, h, style1_seeds, style2_seed, style_ranges):
    style1_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in style1_seeds)
    style2_latents = np.stack(
        np.random.RandomState(style2_seed).randn(Gs.input_shape[1]) for _ in range(len(style_ranges)))
    style1_dlatents = Gs.components.mapping.run(style1_latents, None)  # [seed, layer, component]
    style2_dlatents = Gs.components.mapping.run(style2_latents, None)  # [seed, layer, component]
    style1_images = Gs.components.synthesis.run(style1_dlatents, randomize_noise=False, **synthesis_kwargs)
    style2_image = Gs.components.synthesis.run(style2_dlatents, randomize_noise=False, **synthesis_kwargs)[0]

    canvas = PIL.Image.new('RGB', (w * (len(style_ranges) + 1), h * (len(style1_seeds) + 1)), 'white')

    for row, src_image in enumerate(list(style1_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (0, (row + 1) * h))
    for row in range(len(style_ranges)):
        canvas.paste(PIL.Image.fromarray(style2_image, 'RGB'), ((row + 1) * h, 0))
        for col in range(len(style1_seeds)):
            mixed_dlatent = np.array([style1_dlatents[col]])
            mixed_dlatent[:, style_ranges[row], :] = style2_dlatents[row, style_ranges[row], :]
            image = Gs.components.synthesis.run(mixed_dlatent, randomize_noise=False, **synthesis_kwargs)[0]
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((row + 1) * w, (col + 1) * h))
    canvas.save(png)
    print(png)


def main():
    tflib.init_tf()
    baseline_network_pkl = '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl'
    _G, _D, Gs_baseline = misc.load_pkl(baseline_network_pkl)
    draw_style_mixing_figure_transition('images/style_mix_baseline.png', Gs_baseline, w=256, h=256, style1_seeds=[12, 25, 1], style2_seed=[45], style_ranges=[list(range(i-2, i)) for i in range(2, 16, 2)])

    no_style_mix_network_pkl = '../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-011726.pkl'
    _G, _D, Gs_no_style_mix = misc.load_pkl(no_style_mix_network_pkl)
    draw_style_mixing_figure_transition('images/style_mix_removed.png', Gs_no_style_mix, w=256, h=256, style1_seeds=[10, 56, 1], style2_seed=[34], style_ranges=[list(range(i-2, i)) for i in range(2, 16, 2)])


if __name__ == "__main__":
    main()

