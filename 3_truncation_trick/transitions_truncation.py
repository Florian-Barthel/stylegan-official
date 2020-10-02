import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import training.misc as misc


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


psis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

def transitions(png, w, h, seed):
    col = 0
    canvas = PIL.Image.new('RGB', (w * len(psis), h), 'white')
    baseline_network_pkl = '../../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl'
    G, _D, Gs = misc.load_pkl(baseline_network_pkl)
    for psi in psis:
        latents = np.random.RandomState([seed]).randn(1, Gs.input_shape[1])

        images = Gs.run(latents, None, is_validation=True, randomize_noise=False, truncation_psi_val=psi,
                        truncation_cutoff_val=14)
        images = tflib.convert_images_to_uint8(images).eval()

        images = np.transpose(images, (0, 2, 3, 1))
        canvas.paste(PIL.Image.fromarray(images[0], 'RGB'), (col, 0))

        col += w
    canvas.save(png)


def main():
    tflib.init_tf()
    transitions('tranistions_truncation4.png', 256, 256, 14)
    transitions('tranistions_truncation5.png', 256, 256, 13)
    transitions('tranistions_truncation6.png', 256, 256, 12)
    transitions('tranistions_truncation7.png', 256, 256, 11)



if __name__ == "__main__":
    main()
