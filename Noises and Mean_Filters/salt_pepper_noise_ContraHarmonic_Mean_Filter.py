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


# Define a function for the contraharmonic mean filter
def contraharmonic_mean_filter(image, kernel_size, Q):  # work well for salt-pepper noise
                                                        # +Q: eliminate pepper; -Q: eliminate salt
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.uint8)

    half_kernel_size = kernel_size // 2

    for i in range(half_kernel_size, height - half_kernel_size):
        for j in range(half_kernel_size, width - half_kernel_size):
            neighborhood = image[i - half_kernel_size:i + half_kernel_size + 1,
                           j - half_kernel_size:j + half_kernel_size + 1]
            numerator = np.sum(np.power(neighborhood, Q + 1))
            denominator = np.sum(np.power(neighborhood, Q))
            if denominator != 0:
                result[i, j] = np.clip(numerator / denominator, 0, 255)

    return result


# Filter images with contraharmonic mean filter (Q = 1.5)
contraharmonic_filtered_image_1_5_pepper = contraharmonic_mean_filter(image_with_pepper_noise, 3, 1.5)
contraharmonic_filtered_image_1_5_salt = contraharmonic_mean_filter(image_with_salt_noise, 3, 1.5)

# Filter images with contraharmonic mean filter (Q = -1.5)
contraharmonic_filtered_image_minus_1_5_pepper = contraharmonic_mean_filter(image_with_pepper_noise, 3, -1.5)
contraharmonic_filtered_image_minus_1_5_salt = contraharmonic_mean_filter(image_with_salt_noise, 3, -1.5)

# Create subplots for 8 images in two rows
plt.figure(figsize=(12, 8))

# Row 1
plt.subplot(2, 4, 1), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 2), plt.imshow(image_with_pepper_noise, cmap='gray')
plt.title('Pepper Noise'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 3), plt.imshow(contraharmonic_filtered_image_1_5_pepper, cmap='gray')
plt.title('Contraharmonic Filter (Q=1.5)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 4), plt.imshow(contraharmonic_filtered_image_minus_1_5_pepper, cmap='gray')
plt.title('Contraharmonic Filter (Q=-1.5)'), plt.xticks([]), plt.yticks([])

# Row 2
plt.subplot(2, 4, 5), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 6), plt.imshow(image_with_salt_noise, cmap='gray')
plt.title('Salt Noise'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 7), plt.imshow(contraharmonic_filtered_image_1_5_salt, cmap='gray')
plt.title('Contraharmonic Filter (Q=1.5)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 4, 8), plt.imshow(contraharmonic_filtered_image_minus_1_5_salt, cmap='gray')
plt.title('Contraharmonic Filter (Q=-1.5)'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
