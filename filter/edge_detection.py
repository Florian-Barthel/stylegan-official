import cv2
from matplotlib import pyplot as plt

x = 7
y = 1

img = cv2.imread('../result_images/fakes008086.png', 0)
crop_img = img[256 * y:256 * y + 256, 256 * x:256 * x + 256]
edges = cv2.Sobel(crop_img, cv2.CV_64F, 1, 1, ksize=3)

plt.subplot(121), plt.imshow(crop_img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Sobel Filtered (X + Y)'), plt.xticks([]), plt.yticks([])

plt.show()
