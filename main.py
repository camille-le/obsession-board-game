from matplotlib import pyplot as plt
import cv2
import numpy as np

# Read in Image
img = cv2.imread('scan1.jpeg')
assert img is not None, "file could not be read, check file path."

# Get Gray Image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get Blurred Image
kernel_size = (11, 11)
blurred_img = cv2.GaussianBlur(gray_img, kernel_size, 0)

# 1. FIX: Apply Canny Edge Detection to the blurred image (not the original color image)
edges = cv2.Canny(blurred_img, 100, 200)
cv2.waitKey(0)
cv2.imshow("OK...", edges)


# Matplotlib plot section (kept for completeness, but not required for cropping)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(rgb_img)
plt.title('Original Color Image')
plt.xticks([])
plt.yticks([])
plt.subplot(132)
plt.imshow(gray_img, cmap='gray')
plt.title("Gray Image")
plt.xticks([])
plt.yticks([])
plt.subplot(133)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([])
plt.yticks([])
plt.show()