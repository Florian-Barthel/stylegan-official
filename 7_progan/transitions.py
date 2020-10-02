import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import training.misc as misc


synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

_Gs_cache = dict()


psis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

def transitions(png, w, h, seed1, seed2, pkl):
    col = 0
    canvas = PIL.Image.new('RGB', (h, w * len(psis)), 'white')
    G, _D, Gs = misc.load_pkl(pkl)
    for psi in psis:
        latent1 = np.random.RandomState([seed1]).randn(1, Gs.input_shape[1])
        latent2 = np.random.RandomState([seed2]).randn(1, Gs.input_shape[1])
        dlatent1 = Gs.components.mapping.get_output_for(latent1, None, is_validation=True)
        dlatent2 = Gs.components.mapping.get_output_for(latent2, None, is_validation=True)
        dlatent_int = psi * dlatent1 + (1 - psi) * dlatent2
        images = Gs.components.synthesis.get_output_for(dlatent_int, is_validation=True, randomize_noise=True)
        images = tflib.convert_images_to_uint8(images).eval()

        images = np.transpose(images, (0, 2, 3, 1))
        canvas.paste(PIL.Image.fromarray(images[0], 'RGB'), (0, col))

        col += w
    canvas.save(png)


def main():
    tflib.init_tf()
    baseline_network_pkl = '../../results/00015-sgan-ffhq256-1gpu-baseline/network-snapshot-014526.pkl'
    no_progan_network_pkl = '../../results/00001-sgan-ffhq256-2gpu-remove-progan/network-snapshot-014800.pkl'

    transitions('tranistions_baseline_1.png', 256, 256, 14, 5, baseline_network_pkl)
    # transitions('tranistions_no_progan_1.png', 256, 256, 7, 5, no_progan_network_pkl)
    #
    # transitions('tranistions_baseline_2.png', 256, 256, 1, 2, baseline_network_pkl)
    # transitions('tranistions_no_progan_2.png', 256, 256, 1, 2, no_progan_network_pkl)
    #
    # transitions('tranistions_baseline_3.png', 256, 256, 3, 4, baseline_network_pkl)
    # transitions('tranistions_no_progan_3.png', 256, 256, 3, 4, no_progan_network_pkl)
    #
    # transitions('tranistions_baseline_4.png', 256, 256, 6, 8, baseline_network_pkl)
    # transitions('tranistions_no_progan_4.png', 256, 256, 6, 8, no_progan_network_pkl)

    # transitions('tranistions_baseline_5.png', 256, 256, 5, 6, baseline_network_pkl)
    # transitions('tranistions_no_progan_5.png', 256, 256, 3, 7, no_progan_network_pkl)

    # transitions('tranistions_no_progan_6.png', 256, 256, 21, 14, no_progan_network_pkl)
    # transitions('tranistions_no_progan_7.png', 256, 256, 22, 14, no_progan_network_pkl)
    # transitions('tranistions_no_progan_8.png', 256, 256, 23, 14, no_progan_network_pkl)
    # transitions('tranistions_no_progan_10.png', 256, 256, 25, 14, no_progan_network_pkl)
    # transitions('tranistions_no_progan_12.png', 256, 256, 113, 14, no_progan_network_pkl)
    # transitions('tranistions_no_progan_13.png', 256, 256, 114, 14, no_progan_network_pkl)
    # transitions('tranistions_no_progan_14.png', 256, 256, 115, 14, no_progan_network_pkl)
    # transitions('tranistions_no_progan_15.png', 256, 256, 116, 14, no_progan_network_pkl)
    # transitions('tranistions_no_progan_16.png', 256, 256, 117, 14, no_progan_network_pkl)
    #transitions('tranistions_no_progan_17.png', 256, 256, 118, 14, no_progan_network_pkl)


if __name__ == "__main__":
    main()
