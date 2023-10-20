import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh gốc
from docutils.nodes import image

image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)

# Thêm salt noise
def add_salt_noise(image, salt_prob):
    noisy_image = np.copy(image)
    salt_coords = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_coords] = 255
    return noisy_image

salt_prob = 0.05  # Xác suất thêm salt noise cho mỗi pixel
salt_image = add_salt_noise(image, salt_prob)

# Thêm pepper noise
def add_pepper_noise(image, pepper_prob):
    noisy_image = np.copy(image)
    pepper_coords = np.random.rand(*image.shape) < pepper_prob
    noisy_image[pepper_coords] = 0
    return noisy_image

pepper_prob = 0.05  # Xác suất thêm pepper noise cho mỗi pixel
pepper_image = add_pepper_noise(image, pepper_prob)

# Sử dụng bộ lọc max và min để giảm nhiễu
# Max: good for bright area (salt)
# Min: good for dark (pepper)
max_filtered_image = cv2.erode(salt_image, np.ones((3, 3), np.uint8))
min_filtered_image = cv2.dilate(pepper_image, np.ones((3, 3), np.uint8))

max_filter_for_pepper = cv2.dilate(pepper_image, np.ones((3, 3), np.uint8))
min_filter_for_salt = cv2.erode(salt_image, np.ones((3, 3), np.uint8))

# plt.figure("Figure 1")

# Hiển thị hình ảnh gốc và hình ảnh đã xử lý
plt.subplot(241), plt.imshow(image, cmap='gray')
plt.title('Hình ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(242), plt.imshow(salt_image, cmap='gray')
plt.title('Hình ảnh với salt noise'), plt.xticks([]), plt.yticks([])

plt.subplot(243), plt.imshow(max_filtered_image, cmap='gray')
plt.title('Hình ảnh đã xử lý bằng max filter'), plt.xticks([]), plt.yticks([])

plt.subplot(244), plt.imshow(max_filter_for_pepper, cmap='gray')
plt.title('Max filter cho pepper noise'), plt.xticks([]), plt.yticks([])

# plt.show()

plt.subplot(245), plt.imshow(image, cmap='gray')
plt.title('Hình ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(246), plt.imshow(pepper_image, cmap='gray')
plt.title('Hình ảnh với pepper noise'), plt.xticks([]), plt.yticks([])

plt.subplot(247), plt.imshow(min_filtered_image, cmap='gray')
plt.title('Hình ảnh đã xử lý bằng min filter'), plt.xticks([]), plt.yticks([])

plt.subplot(248), plt.imshow(min_filter_for_salt, cmap='gray')
plt.title('Min filter cho salt noise'), plt.xticks([]), plt.yticks([])

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
