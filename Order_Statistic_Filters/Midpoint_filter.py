import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh gốc
image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)

# Thêm nhiễu Gaussian
mean = 0
variance = 100  # Điều chỉnh độ nhiễu tại đây
stddev = np.sqrt(variance)
gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, gaussian_noise)

# Kích thước cửa sổ hoặc kernel (ví dụ: 3x3)
window_size = 3
# Padding ảnh để tránh việc truy cập index ngoài biên
padded_image = cv2.copyMakeBorder(noisy_image, window_size // 2, window_size // 2, window_size // 2, window_size // 2, cv2.BORDER_CONSTANT)
# Hàm Midpoint Filter
def midpoint_filter(image, window_size):  # Good for randomly distributed noise (Gaussian, Uniform)
    height, width = image.shape
    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            # Trích xuất cửa sổ/kernel từ hình ảnh gốc
            window = padded_image[i:i + window_size, j:j + window_size]
            max_value = np.max(window)
            min_value = np.min(window)

            # Tính trung vị (midpoint) của cực đại và cực tiểu
            midpoint = (max_value + min_value) // 2

            result[i, j] = midpoint

    return result

# Áp dụng Midpoint Filter
filtered_image = midpoint_filter(noisy_image, window_size)

# Hiển thị hình ảnh gốc, hình ảnh đã xử lý, và histogram của hình ảnh đã xử lý
plt.figure(figsize=(10, 5))

plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Hình ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(noisy_image, cmap='gray')
plt.title('Hình ảnh với nhiễu Gaussian'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(filtered_image, cmap='gray')
plt.title('Hình ảnh đã xử lý bằng Midpoint Filter'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
