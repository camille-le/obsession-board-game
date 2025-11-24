import numpy as np
import cv2

# Read in Image
filename = 'Scan 3.jpeg'

img = cv2.imread(filename)

if img is None:
    print(f"Error: Could not load image {filename}")
    exit()

# --- Initial Processing (Unchanged) ---
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
kernel_size = (11, 11)
blurred_img = cv2.GaussianBlur(gray_img, kernel_size, 0)
thresh = cv2.adaptiveThreshold(
    blurred_img,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    10
)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
MIN_AREA = 500
filtered_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]

# --- Circular Crop with Alpha Channel ---

if filtered_contours:
    largest_contour = max(filtered_contours, key=cv2.contourArea)

    # 1. Get the bounding box coordinates (x, y, w, h)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # 2. Crop the original color image to the bounding box
    initial_crop = img[y:y + h, x:x + w]
    crop_h, crop_w = initial_crop.shape[:2]

    # Define a margin to cut off the brown line/edge boundary
    MARGIN_PIXELS = 5
    center_x, center_y = crop_w // 2, crop_h // 2
    radius = min(center_x, center_y) - MARGIN_PIXELS
    if radius <= 0:
        radius = 1

    # 3. Create the white circular mask (This will become the Alpha Channel)
    mask = np.zeros((crop_h, crop_w), dtype="uint8")
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    # 4. Add the Alpha Channel to the initial crop
    # cv2.split breaks the BGR image into 3 separate single-channel arrays
    b, g, r = cv2.split(initial_crop)

    # cv2.merge combines the B, G, R channels with the mask as the 4th (Alpha) channel
    # Where the mask is white (255), the image will be opaque.
    # Where the mask is black (0), the image will be transparent.
    bgr_alpha_image = cv2.merge([b, g, r, mask])

    # 5. Final crop (slicing) to remove the black padding outside the circle's bounding square
    final_x = center_x - radius
    final_y = center_y - radius
    side_length = 2 * radius

    # The final cropped image with a transparent background in the corners
    cropped_img = bgr_alpha_image[final_y:final_y + side_length, final_x:final_x + side_length]

    kernel_size = (7, 7)
    smoothed_img = cv2.GaussianBlur(cropped_img, kernel_size, 0)

    # --- Save the final image as PNG (necessary for transparency) ---
    output_filename = filename.split('.')[0] + " Transparent.png"
    cv2.imwrite(output_filename, smoothed_img)

    # NOTE: cv2.imshow cannot fully display transparency, but we show the result anyway.
    # cv2.imshow("Final Circular Cropped Image (Corners are Transparent in saved PNG)", cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(f"✅ Image successfully cropped and saved with a transparent background.")
    print(f"The cropped result is a {side_length}x{side_length} PNG file saved to: **{output_filename}**")
else:
    print("Nothing to crop")


# ... (Rest of the original code/comments)

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
