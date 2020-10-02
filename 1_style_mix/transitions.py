import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import training.misc as misc


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


baseline_network_pkl = ['../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-010526.pkl',
                        '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-010926.pkl',
                        '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-011326.pkl',
                        '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-011726.pkl',
                        '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-012126.pkl',
                        '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-012526.pkl',
                        '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-012926.pkl']

no_style_network_pkl = ['../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-010526.pkl',
                        '../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-010926.pkl',
                        '../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-011326.pkl',
                        '../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-011726.pkl',
                        '../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-012126.pkl',
                        '../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-012526.pkl',
                        '../results/00023-sgan-ffhq256-2gpu-remove-style-mix/network-snapshot-012926.pkl']

def transitions(png, w, h, pkls, seed):
    col = 0
    canvas = PIL.Image.new('RGB', (w * 7, h), 'white')
    for pkl in pkls:
        G, _D, Gs = misc.load_pkl(pkl)
        latents = np.random.RandomState([seed]).randn(Gs.input_shape[1])

        dlatents = Gs.components.mapping.run(np.stack([latents]), None)
        image = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)[0]
        canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col, 0))

        col += w
    canvas.save(png)


def main():
    tflib.init_tf()
    transitions('images/tranistions_no_style_1.png', 256, 256, no_style_network_pkl, 12)
    transitions('images/tranistions_no_style_2.png', 256, 256, no_style_network_pkl, 13)
    # transitions('images/tranistions_baseline.png', 256, 256, baseline_network_pkl, 10)


if __name__ == "__main__":
    main()
