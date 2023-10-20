import cv2
import numpy as np
import matplotlib.pyplot as plt

# image = cv2.imread('../Turkey Coffee.png', 1)
image = cv2.imread('../Bright_Future.jpg', cv2.COLOR_BGR2RGB)
# cv2.imshow('', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

kernel = np.ones((5, 5), np.uint8)
# Sử dụng bộ lọc max và min để giảm nhiễu
max_filtered_image = cv2.erode(image, kernel, iterations=1)
min_filtered_image = cv2.dilate(image, kernel, iterations=1)

# Sử dụng median filter để giảm nhiễu
median_filtered_image = cv2.medianBlur(image, 3)  # Kích thước kernel là 3x3

# Chuyển đổi hệ màu của ảnh đã xử lý sang RGB
# max_filtered_image_rgb = cv2.cvtColor(max_filtered_image, cv2.COLOR_RGB2BGR)
# min_filtered_image_rgb = cv2.cvtColor(min_filtered_image, cv2.COLOR_RGB2BGR)

plt.subplot(221), plt.imshow(image)
plt.title('Hình ảnh gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.imshow(max_filtered_image)
plt.title('Hình ảnh đã xử lý bằng max filter'), plt.xticks([]), plt.yticks([])

plt.subplot(223), plt.imshow(min_filtered_image)
plt.title('Hình ảnh đã xử lý bằng min filter'), plt.xticks([]), plt.yticks([])

plt.subplot(224), plt.imshow(min_filtered_image)
plt.title('Hình ảnh đã xử lý bằng median filter'), plt.xticks([]), plt.yticks([])

plt.show()
