import numpy as np
import cv2

# Read in Image
img = cv2.imread('scan1.jpeg')

# Display Image in Window
cv2.imshow("Image", img)
cv2.waitKey(0)

# Changing Color Spaces
# https://docs.opencv.org/3.4/df/d9d/tutorial_py_colorspaces.html

# Change an image to a gray one
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Applying a Blur Effect to the Gray Image
kernel_size = (11, 11)
blurred_img = cv2.GaussianBlur(gray_img, kernel_size, 0)


## Thresholding the Blurred Image
# Use a strong Adaptive Threshold for robust results on scans.
# This makes text/lines stand out crisply against the background.
thresh = cv2.adaptiveThreshold(
    blurred_img,           # Input: The blurred grayscale image
    255,                     # Value to set foreground pixels (white)
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Adaptive method
    cv2.THRESH_BINARY_INV,   # Output: Inverse binary (White objects on Black background)
    31,                      # Block Size (must be odd)
    10                       # C (Constant to subtract)
)

# Display the thresholded image for debugging
cv2.imshow('Thresholded Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()



# RETR_EXTERNAL retrieves only the extreme outer contours (good for finding the card's edge)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define minimum area to filter out noise specks
MIN_AREA = 500  # Adjust this based on your object size

# Filter contours
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > MIN_AREA:
        filtered_contours.append(contour)


## Crop the Original Image to the Largest Contour

if filtered_contours:
    # Get the largest contour (the object's main boundary)
    largest_contour = max(filtered_contours, key=cv2.contourArea)

    # Get the bounding box coordinates (x, y, width, height)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the original color image using NumPy slicing
    # Syntax: image[y: y + h, x: x + w]
    cropped_img = img[y:y + h, x:x + w]

    # Display the final cropped image
    cv2.imshow("Final Cropped Image", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Image successfully cropped to size: {w}x{h}")
else:
    print("No large objects found to crop.")


# --- 3. Display Results ---
# cv2.imshow('Blurred Image', blurred_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


## Canny Edge Detection

# from matplotlib import pyplot as plt
#
# edges = cv2.Canny(img, 100, 200)
#
# plt.subplot(121), plt.imshow(gray_img, cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()
#
