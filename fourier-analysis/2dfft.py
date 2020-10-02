# empfohlene Imports
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2

x = 7
y = 2
fake_img = cv2.imread('../results/00015-sgan-ffhq256-1gpu-baseline/fakes015006.png', 0)
fake_img = fake_img[256 * y:256 * y + 256, 256 * x:256 * x + 256]
plt.imshow(fake_img, cmap=cm.gray)

real_img = cv2.imread('E:/ffhq_256/00005.png', 0)
plt.imshow(real_img, cmap=cm.gray)

def get_fft(bild):
    # wandeln Sie das Bild in ein 'float64' Bild um und berechnen Sie die FFT
    bildf64 = bild.astype('float64')
    fftbild = np.fft.fft2(bildf64)
    print(fftbild.shape)
    print(fftbild.dtype)
    print(fftbild[1, 1])
    
    fftreal = np.real(fftbild)
    fftimaginary = np.imag(fftbild)
    fftmag = np.abs(fftbild)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Real")

    plt.imshow(np.fft.fftshift(np.log(fftreal)))
    plt.subplot(1, 3, 2)
    plt.title("Imaginary")
    plt.imshow(np.fft.fftshift(np.log(fftimaginary)))
    plt.subplot(1, 3, 3)
    plt.title("Magnitude")
    plt.imshow(np.fft.fftshift(np.log(fftmag)))


get_fft(real_img)
get_fft(fake_img)
plt.show()
