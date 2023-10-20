import cv2
import numpy as np
import matplotlib.pyplot as plt

# # Đọc hình ảnh gốc
# original_image = cv2.imread('Gaussian noise.png', cv2.IMREAD_GRAYSCALE)
#
# # Thêm nhiễu Gaussian vào hình ảnh
# mean = 0
# stddev = 25
# image_with_gaussian_noise = original_image + np.random.normal(mean, stddev, original_image.shape).astype(np.uint8)
#
# # Giới hạn giá trị pixel trong khoảng 0 đến 255
# image_with_gaussian_noise = np.clip(image_with_gaussian_noise, 0, 255)

image_with_gaussian_noise = cv2.imread("Gaussian noise 1.png")

# Vẽ histogram cho hình ảnh có nhiễu Gaussian
plt.hist(image_with_gaussian_noise.ravel(), bins=256, range=[0, 256])
plt.title('Histogram for Image with Gaussian Noise')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
