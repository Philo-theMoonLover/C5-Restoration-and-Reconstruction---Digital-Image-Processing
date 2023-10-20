import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc hình ảnh gốc
image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)

# Define function to add salt noise with given probability
def add_salt_noise(image, salt_probability):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_probability * image.size).astype(int)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[tuple(salt_coords)] = 255
    return noisy_image

# Define function to add pepper noise with given probability
def add_pepper_noise(image, pepper_probability):
    noisy_image = np.copy(image)
    num_pepper = np.ceil(pepper_probability * image.size).astype(int)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[tuple(pepper_coords)] = 0
    return noisy_image

# Corrupt the original image with pepper noise (probability=0.1)
image_with_pepper_noise = add_pepper_noise(image, 0.1)

# Corrupt the original image with salt noise (probability=0.1)
noisy_image = add_salt_noise(image_with_pepper_noise, 0.1)

# Thêm nhiễu Gaussian
mean = 0
variance = 100  # Điều chỉnh độ nhiễu tại đây
stddev = np.sqrt(variance)
gaussian_noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
noisy_image = cv2.add(noisy_image, gaussian_noise)

# Kích thước cửa sổ hoặc kernel (ví dụ: 3x3)
window_size = 3

# Tham số alpha (số lượng giá trị bị loại bỏ)
alpha = 3  # Điều chỉnh alpha tại đây

# Padding ảnh để tránh việc truy cập index ngoài biên
padded_image = cv2.copyMakeBorder(noisy_image, window_size // 2, window_size // 2, window_size // 2, window_size // 2, cv2.BORDER_CONSTANT)

# Hàm Alpha-Trimmed Mean Filter
def alpha_trimmed_mean_filter(image, window_size, alpha):
    height, width = image.shape
    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            # Trích xuất cửa sổ/kernel từ hình ảnh gốc
            window = padded_image[i:i + window_size, j:j + window_size]

            # Sắp xếp giá trị pixel trong cửa sổ/kernel
            sorted_values = np.sort(window.ravel())

            # Loại bỏ alpha/2 giá trị pixel từ đầu và cuối danh sách đã sắp xếp
            trimmed_values = sorted_values[alpha // 2:-(alpha // 2)]

            # Tính trung bình cộng của các giá trị pixel còn lại
            result[i, j] = np.mean(trimmed_values)

    return result

# Áp dụng Alpha-Trimmed Mean Filter
filtered_image = alpha_trimmed_mean_filter(noisy_image, window_size, alpha)

plt.figure("Figure 1", figsize=(12, 6))
# Hiển thị hình ảnh gốc, hình ảnh đã thêm nhiễu và hình ảnh đã xử lý bằng median filter
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Hình ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(noisy_image, cmap='gray')
plt.title('Hình ảnh với salt-pepper noise và gaussian noise'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(filtered_image, cmap='gray')
plt.title('Hình ảnh đã xử lý bằng Alpha-Trimmed Mean Filter'), plt.xticks([]), plt.yticks([])
plt.show()