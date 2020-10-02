import PIL.Image
import numpy as np

x = 0
y = 1
res = 256

canvas = PIL.Image.new('RGB', (256 * 6, 256), 'white')

# 8x8
fake_img = PIL.Image.open('../results/00015-sgan-ffhq256-1gpu-baseline/fakes000561.png')
fake_img = np.asarray(fake_img)
fake_img = fake_img[res * y:res * y + res, res * x:res * x + res, :]
canvas.paste(PIL.Image.fromarray(fake_img, 'RGB'), (0, 0))
offset = res

# 16x16

fake_img = PIL.Image.open('../results/00015-sgan-ffhq256-1gpu-baseline/fakes001764.png')
fake_img = np.asarray(fake_img)
fake_img = fake_img[res * y:res * y + res, res * x:res * x + res, :]
canvas.paste(PIL.Image.fromarray(fake_img, 'RGB'), (offset, 0))
offset += res

# 32x32

fake_img = PIL.Image.open('../results/00015-sgan-ffhq256-1gpu-baseline/fakes002965.png')
fake_img = np.asarray(fake_img)
fake_img = fake_img[res * y:res * y + res, res * x:res * x + res, :]
canvas.paste(PIL.Image.fromarray(fake_img, 'RGB'), (offset, 0))
offset += res

# 64x64

fake_img = PIL.Image.open('../results/00015-sgan-ffhq256-1gpu-baseline/fakes004165.png')
fake_img = np.asarray(fake_img)
fake_img = fake_img[res * y:res * y + res, res * x:res * x + res, :]
canvas.paste(PIL.Image.fromarray(fake_img, 'RGB'), (offset, 0))
offset += res

# 128x128

fake_img = PIL.Image.open('../results/00015-sgan-ffhq256-1gpu-baseline/fakes005366.png')
fake_img = np.asarray(fake_img)
fake_img = fake_img[res * y:res * y + res, res * x:res * x + res, :]
canvas.paste(PIL.Image.fromarray(fake_img, 'RGB'), (offset, 0))
offset += res

# 256x256
fake_img = PIL.Image.open('../results/00015-sgan-ffhq256-1gpu-baseline/fakes006006.png')
fake_img = np.asarray(fake_img)
fake_img = fake_img[res * y:res * y + res, res * x:res * x + res, :]
canvas.paste(PIL.Image.fromarray(fake_img, 'RGB'), (offset, 0))
offset += res

canvas.save('growing_res.png')