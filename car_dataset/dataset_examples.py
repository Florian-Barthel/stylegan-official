import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import config
from training import dataset
from dnnlib import EasyDict

dataset_args = EasyDict(tfrecord_dir='cars', resolution=512)

def draw_style_mixing_figure_transition(png):

    n = 8
    w = 512
    h = 512
    canvas = PIL.Image.new('RGB', (w * n, h), 'white')
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **dataset_args)

    image = training_set.get_minibatch_np(1, 0)[0][0].transpose(1, 2, 0)
    image = training_set.get_minibatch_np(1, 0)[0][0].transpose(1, 2, 0)
    res = 8
    for i in range(n):

        pil_image = PIL.Image.fromarray(image, 'RGB')
        pil_image.thumbnail((res, res))
        pil_image.thumbnail((512, 512))
        canvas.paste(pil_image, (i * w, 0))
        res = res * 2
    canvas.save(png)
    print(png)


def main():
    tflib.init_tf()
    draw_style_mixing_figure_transition('images/dataset_examples_1.png')

if __name__ == "__main__":
    main()

