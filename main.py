from matplotlib import pyplot as plt
import cv2

# Read in Image
img = cv2.imread('scan1.jpeg')

# Get Gray Image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get Blurred Image
kernel_size = (11, 11)
blurred_img = cv2.GaussianBlur(gray_img, kernel_size, 0)

# Apply Canny Edge Detection
edges = cv2.Canny(img, 100, 200)

# Plot Before and After
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(15, 5))

# 1st Subplot: Original Image (True Color)
plt.subplot(131)
plt.imshow(rgb_img)
plt.title('Original Color Image')
plt.xticks([]), plt.yticks([])

# 2nd Subplot: Gray Image (Single Channel)
plt.subplot(132)
plt.imshow(gray_img, cmap='gray')
plt.title("Gray Image")
plt.xticks([]), plt.yticks([])

# 3rd Subplot: Edge Image (Binary result must use 'gray' cmap)
plt.subplot(133)
plt.imshow(edges, cmap='gray')
plt.title('Edge Image')
plt.xticks([]), plt.yticks([])

plt.show()

# For better accuracy, use binary images. So before finding contours, apply threshold or canny edge detection.
