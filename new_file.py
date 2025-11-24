import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

filename = 'Scan.jpeg'

# --- Definitions and Image Loading ---

# Define a 3x3 kernel size
kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

# Define a 5x5 kernel size (Not used in the final display crop, but kept for context)
kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
img = cv2.fastNlMeansDenoising(img, None, h=10, templateWindowSize=7, searchWindowSize=21)

if img is None:
    print("Error: Image not loaded. Check file path.")
    exit()

# --- Filter Application ---
k3_img = ndimage.convolve(img, kernel_3x3)
k5_img = ndimage.convolve(img, kernel_5x5)

# Calculate the differential HPF (hpf_img)
blurred_img = cv2.GaussianBlur(img, (17, 17), 0)
hpf_img = img - blurred_img

# 2. Apply a threshold to create a mask of the non-background area
#    We use a low threshold (e.g., 20) to detect areas where the filter output is significant.
_, thresh = cv2.threshold(hpf_img, 20, 255, cv2.THRESH_BINARY)

# 3. Find the largest contour (the filter area)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

x, y, w, h = 0, 0, 0, 0
max_area = 0

for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        x, y, w, h = cv2.boundingRect(contour)

# 4. Apply a safe padding and crop the ORIGINAL image
# The coordinates (x, y, w, h) were derived from the HPF analysis,
# but they are used to crop the original image (img) for clarity.
padding = 15  # Add 15 pixels of border

if w > 0 and h > 0:
    # Calculate padded coordinates, clamping to image boundaries
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)

    # Crop the ORIGINAL image
    cropped_img = img[y_start:y_end, x_start:x_end]
    cv2.imwrite("cropped_scan1.jpeg", img)

    # Also crop the hpf_img for comparison
    hpf_img_cropped = hpf_img[y_start:y_end, x_start:x_end]
    cv2.imwrite("cropped_orig.jpeg", hpf_img_cropped)

    # Convert cropped HPF to displayable format
    hpf_img_cropped_display = cv2.normalize(hpf_img_cropped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print("HI DONE")
else:
    print("Warning: No significant filter area found for cropping. Displaying full image.")
    cropped_img = img
    hpf_img_cropped_display = cv2.normalize(hpf_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# --- Define the data for plotting ---
# Use the cropped original image and the cropped HPF image for the final display
plots = [(cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR), 'Cropped Original', None),
         # Original (cropped) is BGR for compatibility
         (cv2.cvtColor(k3_img, cv2.COLOR_GRAY2BGR), '3x3 (Full)', None),  # Keeping these as full-size for context
         (cv2.cvtColor(blurred_img, cv2.COLOR_GRAY2BGR), 'Blurred (Full)', None),
         (hpf_img_cropped_display, "HPF (Cropped)", 'gray')]

# --- Create figure and subplots ---
# fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#
# # Plot the images and set titles using iteration
# for ax, (img, title, cmap) in zip(axes, plots):
#     # Ensure images are properly scaled for display if they are not 3-channel
#     if len(img.shape) == 2:
#         ax.imshow(img, cmap=cmap)
#     else:
#         ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#     ax.set_title(title)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# plt.tight_layout()
# plt.show()