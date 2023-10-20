import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
original_image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)


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
image_with_pepper_noise = add_pepper_noise(original_image, 0.1)

# Corrupt the original image with salt noise (probability=0.1)
image_with_salt_noise = add_salt_noise(original_image, 0.1)


# Định nghĩa hàm bộ lọc trung bình harmonic
def harmonic_mean_filter(image, kernel_size):  # work well for salt(255), fail for pepper(0)
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            block = image[i:i + kernel_size, j:j + kernel_size]
            reciprocal_block = 1.0 / (block.astype(np.float32) + 1e-6)
            result[i:i + kernel_size, j:j + kernel_size] = kernel_size * kernel_size / np.sum(reciprocal_block)

    return result.astype(np.uint8)

# Áp dụng bộ lọc trung bình harmonic để loại bỏ nhiễu
filtered_image_pepper = harmonic_mean_filter(image_with_pepper_noise, 3)  # Sử dụng kernel 3x3 (có thể thay đổi kích thước kernel)

# Áp dụng bộ lọc trung bình harmonic để loại bỏ nhiễu
filtered_image_salt = harmonic_mean_filter(image_with_salt_noise, 3)  # Sử dụng kernel 3x3 (có thể thay đổi kích thước kernel)

# Create subplots for 8 images in two rows
plt.figure(figsize=(12, 8))

# Row 1
plt.subplot(2, 3, 1), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 2), plt.imshow(image_with_pepper_noise, cmap='gray')
plt.title('Pepper Noise'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 3), plt.imshow(filtered_image_pepper, cmap='gray')
plt.title('Filtered Image (Harmonic Mean)'), plt.xticks([]), plt.yticks([])

# Row 2
plt.subplot(2, 3, 4), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 5), plt.imshow(image_with_salt_noise, cmap='gray')
plt.title('Salt Noise'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 6), plt.imshow(filtered_image_salt, cmap='gray')
plt.title('Filtered Image (Harmonic Mean)'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
