import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the original image
original_image = cv2.imread('../circuit board.png', cv2.IMREAD_GRAYSCALE)

# Create Gaussian noise with zero mean and variance of 400
mean = 0
variance = 400
stddev = np.sqrt(variance)
gaussian_noise = np.random.normal(mean, stddev, original_image.shape).astype(np.uint8)

# Add the Gaussian noise to the original image
image_with_gaussian_noise = cv2.add(original_image, gaussian_noise)

# Display the original image and the image with Gaussian noise
plt.subplot(121), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.imshow(image_with_gaussian_noise, cmap='gray')
plt.title('Image with Gaussian Noise'), plt.xticks([]), plt.yticks([])
plt.show()

plt.hist(image_with_gaussian_noise.ravel(), bins=256, range=[0, 256])
plt.title('Histogram for Gaussian Noise')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.show()
