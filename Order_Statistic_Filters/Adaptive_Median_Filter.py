import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh gốc
image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)

# Hàm thêm nhiễu salt và pepper vào hình ảnh
def add_salt_noise(image, salt_probability):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_probability * image.size).astype(int)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255
    return noisy_image

def add_pepper_noise(image, pepper_probability):
    noisy_image = np.copy(image)
    num_pepper = np.ceil(pepper_probability * image.size).astype(int)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[tuple(pepper_coords)] = 0
    return noisy_image

# Thêm nhiễu salt và pepper vào hình ảnh
salt_probability = 0.02  # Đổi xác suất tùy ý
pepper_probability = 0.02  # Đổi xác suất tùy ý
noisy_image = add_salt_noise(image, salt_probability)
noisy_image = add_pepper_noise(noisy_image, pepper_probability)
# Ảnh đã được thêm 2% salt và 2% pepper

# Kích thước cửa sổ hoặc kernel (ví dụ: 3x3)
window_size = 3
# Hàm để kiểm tra xem cửa sổ có cần mở rộng hay không
def is_window_expand(window, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size should be an odd number.")

    median = np.median(window)
    min_val = np.min(window)
    max_val = np.max(window)

    if min_val < median < max_val:
        return False
    return True

# Padding ảnh để tránh việc truy cập index ngoài biên
padded_image = cv2.copyMakeBorder(noisy_image, window_size // 2, window_size // 2, window_size // 2, window_size // 2,
                                  cv2.BORDER_CONSTANT)

# Hàm Adaptive Median Filter
def adaptive_median_filter(image, window_size):
    height, width = image.shape
    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            # Trích xuất cửa sổ/kernel từ hình ảnh gốc
            window = padded_image[i:i + window_size, j:j + window_size]

            # Dòng mã này sẽ kiểm tra và mở rộng cửa sổ nếu cần
            while is_window_expand(window, window_size):
                window_size += 2
                window = padded_image[i:i + window_size, j:j + window_size]

            # Tìm giá trị trung vị trong cửa sổ đã mở rộng
            result[i, j] = np.median(window)
    return result

# Áp dụng Adaptive Median Filter
filtered_image = adaptive_median_filter(noisy_image, window_size)

# Plot
plt.figure("Figure 1", figsize=(12, 6))
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Hình ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(noisy_image, cmap='gray')
plt.title('Hình ảnh với salt và pepper noise'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(filtered_image, cmap='gray')
plt.title('Hình ảnh đã xử lý bằng Adaptive Median Filter'), plt.xticks([]), plt.yticks([])

plt.show()
