import numpy as np
import cv2

# Read in Image
filename = 'Scan.jpeg'

img = cv2.imread(filename)

# Ensure the image was loaded correctly
if img is None:
    print(f"Error: Could not load image {filename}")
    exit()

# --- 1. Basic Setup & Pre-processing (Kept for Context) ---
# You can remove the display windows if you don't need to see the intermediate steps.
# cv2.imshow("Image", img)
# cv2.waitKey(0)

# Changing Color Spaces
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## Applying a Blur Effect to the Gray Image
kernel_size = (11, 11)
blurred_img = cv2.GaussianBlur(gray_img, kernel_size, 0)

## Thresholding (Kept for original contour finding logic, but not strictly needed for just circular crop)
thresh = cv2.adaptiveThreshold(
    blurred_img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    10
)
# cv2.imshow('Thresholded Image', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --- 2. Circular Center Crop using Masking ---

# Get dimensions of the original image
(h, w) = img.shape[:2]

# Define the center and radius for the perfect circle
# The center is simply the middle of the image
center_x, center_y = w // 2, h // 2

# The radius should be the smaller of the two dimensions (width or height) divided by 2,
# so the circle fits perfectly inside the image without cropping the edges.
radius = min(center_x, center_y)

# Create a black mask image of the same size as the original image (grayscale)
# This will be all black (value 0).
mask = np.zeros((h, w), dtype="uint8")

# Draw a white circle (value 255) on the black mask
# This white area defines the region we want to keep.
cv2.circle(mask, (center_x, center_y), radius, 255, -1)

# Apply the mask to the original image using bitwise_and
# This operation keeps the pixels from 'img' only where the 'mask' pixel is white (255).
# The result has the circular region of 'img' and black everywhere else.
masked_img = cv2.bitwise_and(img, img, mask=mask)

# To display the circular crop without the black background, we need to find the bounding box
# of the white circle in the mask and crop the 'masked_img' to that box.

# Find the bounding box of the circle (it's essentially a square with side = 2*radius)
# The top-left corner is (center_x - radius, center_y - radius)
# The bottom-right corner is (center_x + radius, center_y + radius)
x = center_x - radius
y = center_y - radius
crop_w = 2 * radius
crop_h = 2 * radius

# Crop the masked image to this bounding square
circular_cropped_img = masked_img[y:y + crop_h, x:x + crop_w]

# --- 3. Display Results ---

cv2.imshow("Circular Cropped Image", circular_cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Image successfully cropped to a circular area with radius: {radius}")

# Original contour finding and cropping logic is removed/commented out here
# ...