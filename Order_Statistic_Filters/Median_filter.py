import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh gốc
image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)

# Thêm salt noise
def add_salt_noise(image, salt_prob):
    noisy_image = np.copy(image)
    salt_coords = np.random.rand(*image.shape) < salt_prob
    noisy_image[salt_coords] = 255
    return noisy_image

salt_prob = 0.05  # Xác suất thêm salt noise cho mỗi pixel
noisy_image = add_salt_noise(image, salt_prob)

# Thêm pepper noise
def add_pepper_noise(image, pepper_prob):
    noisy_image = np.copy(image)
    pepper_coords = np.random.rand(*image.shape) < pepper_prob
    noisy_image[pepper_coords] = 0
    return noisy_image

pepper_prob = 0.05  # Xác suất thêm pepper noise cho mỗi pixel
noisy_image = add_pepper_noise(noisy_image, pepper_prob)

# Sử dụng median filter để giảm nhiễu
median_filtered_image = cv2.medianBlur(noisy_image, 3)  # Kích thước kernel là 3x3
# Median filter giữ lại các cạnh và chi tiết tốt hơn các bộ lọc tuyến tính

# Sử dụng bộ lọc max và min để giảm nhiễu
max_filtered_image = cv2.erode(noisy_image, np.ones((3, 3), np.uint8))
min_filtered_image = cv2.dilate(noisy_image, np.ones((3, 3), np.uint8))

plt.figure("Figure 1", figsize=(12, 6))
# Hiển thị hình ảnh gốc, hình ảnh đã thêm nhiễu và hình ảnh đã xử lý bằng median filter
plt.subplot(231), plt.imshow(image, cmap='gray')
plt.title('Hình ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.imshow(noisy_image, cmap='gray')
plt.title('Hình ảnh với salt và pepper noise'), plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.imshow(median_filtered_image, cmap='gray')
plt.title('Hình ảnh đã xử lý bằng median filter'), plt.xticks([]), plt.yticks([])

# plt.subplot(234), plt.imshow(image, cmap='gray')
# plt.title('Hình ảnh gốc'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(235), plt.imshow(max_filtered_image, cmap='gray')
# plt.title('Hình ảnh đã xử lý bằng max filter'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(236), plt.imshow(min_filtered_image, cmap='gray')
# plt.title('Hình ảnh đã xử lý bằng min filter'), plt.xticks([]), plt.yticks([])

plt.show()
