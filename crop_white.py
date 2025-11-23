import cv2
import numpy as np


def crop_robustly(image_path, output_path, threshold_val=220, padding=15):
    """
    A more robust cropping function with adjusted parameters and blur.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    # 1. Blur to reduce noise and smooth the background
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur

    # 2. Adjusted Thresholding
    # Lowered the threshold to 220 (from 240) to handle slightly grayish white backgrounds.
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # 3. Morphological Operations (Closing to fill small holes in the object)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 4. Find Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No contours found. Try lowering the 'threshold_val'.")
        return

    # 5. Filter contours by size and find the largest
    min_area = img.shape[0] * img.shape[1] * 0.01  # Require contour to be at least 1% of image area

    # Filter out very small contours (noise)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    if not large_contours:
        print("Error: Only small noise contours found. Try lowering 'min_area'.")
        return

    # Find the largest contour among the filtered list
    largest_contour = max(large_contours, key=cv2.contourArea)

    # 6. Get Bounding Rectangle and Apply Padding
    x, y, w, h = cv2.boundingRect(largest_contour)

    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img.shape[1], x + w + padding)
    y_end = min(img.shape[0], y + h + padding)

    # 7. Crop and Save
    cropped_img = img[y_start:y_end, x_start:x_end]
    cv2.imwrite(output_path, cropped_img)
    print(f"Successfully cropped and saved image to {output_path}")


# --- Configuration ---
INPUT_FILE = 'scan1.jpeg'
OUTPUT_FILE = 'cropped_robust.jpeg'
CUSTOM_THRESHOLD = 220  # Try 200, 210, 220, or 230
PADDING_PIXELS = 15

# --- Run the function ---
if __name__ == '__main__':
    crop_robustly(INPUT_FILE, OUTPUT_FILE, threshold_val=CUSTOM_THRESHOLD, padding=PADDING_PIXELS)