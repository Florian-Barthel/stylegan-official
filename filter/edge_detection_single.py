import cv2
from matplotlib import pyplot as plt

x = 7
y = 1

img = cv2.imread('../result_images/fakes008086.png', 0)
crop_img = img[256 * y:256 * y + 256, 256 * x:256 * x + 256]

noise_img = cv2.imread('kernel_visualization/noise.png', 0)
# noise_edges = (cv2.Sobel(noise_img, cv2.CV_64F, 1, 1, ksize=3) + 1) * 255
noise_laplacian = cv2.Laplacian(noise_img,cv2.CV_64F)


normal_img = cv2.imread('kernel_visualization/water_droplet.png', 0)
# normal_edges = (cv2.Sobel(normal_img, cv2.CV_64F, 1, 1, ksize=3) + 1) * 255
normal_laplacian = cv2.Laplacian(normal_img,cv2.CV_64F)

plt.subplot(121), plt.imshow(normal_laplacian, cmap='gray')
plt.title('Baseline'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(noise_laplacian, cmap='gray')
plt.title('Without Noise'), plt.xticks([]), plt.yticks([])

plt.show()
