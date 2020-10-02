import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import training.misc as misc


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


def transitions(png, w, h):
    baseline_network_pkl = ['../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-007326.pkl',
                            '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-008526.pkl',
                            '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-009726.pkl',
                            '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-010926.pkl',
                            '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-012126.pkl',
                            '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-013326.pkl',
                            '../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl']

    col = 0
    canvas = PIL.Image.new('RGB', (w * 7, h * 2), 'white')
    for pkl in baseline_network_pkl:
        G, _D, Gs = misc.load_pkl(pkl)
        latents = np.random.RandomState([45]).randn(Gs.input_shape[1])

        dlatents = Gs.components.mapping.run(np.stack([latents]), None)
        image = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)[0]
        canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col, 0))

        dlatents = G.components.mapping.run(np.stack([latents]), None)
        image = G.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)[0]
        canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col, h))
        col += w
    canvas.save(png)


def main():
    tflib.init_tf()
    transitions('tranistions_baseline.png', w=256, h=256)


if __name__ == "__main__":
    main()
