import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# Đọc hình ảnh gốc
image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)

# Thêm nhiễu salt và pepper vào hình ảnh
salt_probability = 0.02  # Đổi xác suất tùy ý
pepper_probability = 0.02  # Đổi xác suất tùy ý
noisy_image = add_salt_noise(image, salt_probability)
noisy_image = add_pepper_noise(noisy_image, pepper_probability)
# Ảnh đã được thêm 2% salt và 2% pepper

# Vẽ histogram cho hình ảnh có nhiễu
plt.hist(noisy_image.ravel(), bins=256, range=(0, 255), density=True, color='gray', alpha=0.7)
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.title('Histogram of Noisy Image')
plt.show()

