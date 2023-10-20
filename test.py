# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Hàm thêm nhiễu salt và pepper vào hình ảnh
# def add_salt_noise(image, salt_probability):
#     noisy_image = np.copy(image)
#     num_salt = np.ceil(salt_probability * image.size).astype(int)
#     salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
#     noisy_image[tuple(salt_coords)] = 255
#     return noisy_image
#
# def add_pepper_noise(image, pepper_probability):
#     noisy_image = np.copy(image)
#     num_pepper = np.ceil(pepper_probability * image.size).astype(int)
#     pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
#     noisy_image[tuple(pepper_coords)] = 0
#     return noisy_image
#
# # Đọc hình ảnh gốc
# image = cv2.imread('slide-thank-you-5.png', cv2.IMREAD_GRAYSCALE)
#
# # Thêm nhiễu salt và pepper vào hình ảnh
# salt_probability = 0.02  # Đổi xác suất tùy ý
# pepper_probability = 0.02  # Đổi xác suất tùy ý
# noisy_image = add_salt_noise(image, salt_probability)
# noisy_image = add_pepper_noise(noisy_image, pepper_probability)
# # Ảnh đã được thêm 2% salt và 2% pepper
#
# # Vẽ histogram cho hình ảnh có nhiễu
# # Row 1
# plt.subplot(1, 1, 1), plt.imshow(noisy_image, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.show()


from scipy.datasets import face
from scipy.signal import wiener
import matplotlib.pyplot as plt
import numpy as np
rng = np.random.default_rng()
img = rng.random((40, 40))    #Create a random image
filtered_img = wiener(img, (5, 5))  #Filter the image
f, (plot1, plot2) = plt.subplots(1, 2)
plot1.imshow(img)
plot2.imshow(filtered_img)
plt.show()
