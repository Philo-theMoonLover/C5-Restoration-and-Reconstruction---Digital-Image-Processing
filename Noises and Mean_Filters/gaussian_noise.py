import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
original_image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)
# original_image = cv2.imread('human_01.png', cv2.IMREAD_GRAYSCALE)

# Create Gaussian noise with zero mean and variance of 400
mean = 0
variance = 400
stddev = np.sqrt(variance)
gaussian_noise = np.random.normal(mean, stddev, original_image.shape).astype(np.uint8)

# Add the Gaussian noise to the original image
image_with_gaussian_noise = cv2.add(original_image, gaussian_noise)

# Define a 3x3 kernel for ARITHMETIC MEAN filter
# arithmetic_kernel = np.full((3, 4), 1) / 12
# print(arithmetic_kernel)
# Filter the noisy image with the arithmetic mean filter
arithmetic_kernel = np.ones((3, 3), np.float32) / 9
arithmetic_filtered_image = cv2.filter2D(image_with_gaussian_noise, -1, arithmetic_kernel)


# Define a function for the GEOMETRIC MEAN filter
def geometric_mean_filter(image, kernel_size):
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(height - kernel_size + 1):
        for j in range(width - kernel_size + 1):
            # Tại mỗi vùng cửa sổ, hàm lấy một khối (block) hình ảnh có kích thước kernel_size x kernel_size (3x3)
            block = image[i:i + kernel_size, j:j + kernel_size]
            # Sau đó, nó tính tích (product) của tất cả các pixel trong khối và tính lũy thừa bậc 1/mn
            result[i:i + kernel_size, j:j + kernel_size] = np.prod(block.astype(np.float32)) ** (
                        1.0 / (kernel_size * kernel_size))

    return result.astype(np.uint8)


# Filter the noisy image with the geometric mean filter
geometric_filtered_image = geometric_mean_filter(image_with_gaussian_noise, 3)

# Display the original image, the image with Gaussian noise, and the filtered images
plt.subplot(141), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(142), plt.imshow(image_with_gaussian_noise, cmap='gray')
plt.title('Image with Gaussian Noise'), plt.xticks([]), plt.yticks([])

plt.subplot(143), plt.imshow(arithmetic_filtered_image, cmap='gray')
plt.title('Arithmetic Mean Filtered Image'), plt.xticks([]), plt.yticks([])

plt.subplot(144), plt.imshow(geometric_filtered_image, cmap='gray')
plt.title('Geometric Mean Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
