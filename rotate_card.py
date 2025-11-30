import cv2
import numpy as np
import imutils
import os
# Ensure you have Pillow installed: pip install Pillow
from PIL import Image, ImageDraw


def order_points(pts):
    """
    Orders a list of 4 points in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]  # Bottom-right (largest sum)
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]  # Top-right (smallest difference: y-x)
    rect[3] = pts[np.argmax(d)]  # Bottom-left (largest difference: y-x)
    return rect


def four_point_transform(image, pts):
    """
    Applies perspective transform to a region defined by 4 points to get a top-down view.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate width and height using Euclidean distance
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Define the destination points for the perfect rectangle
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def process_card_image(image_path):
    """
    Uses OpenCV to find the card edges and straighten the perspective.
    Returns the path to the temporary aligned PNG file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        print("[ERROR] Could not find a 4-point contour (card) in the image.")
        return None

    warped = four_point_transform(image, screenCnt.reshape(4, 2))

    # *** FIX 1: Must use PNG for transparency support in the next step ***
    output_path = "aligned_card.png"
    cv2.imwrite(output_path, warped)
    print(f"[INFO] Aligned card saved to {output_path}")
    return output_path


def round_corners(image_path, radius):
    """
    Applies rounded, transparent corners to an input image using Pillow.
    """
    # Image must support RGBA (Red, Green, Blue, Alpha/Transparency)
    image = Image.open(image_path).convert("RGBA")

    # Create a mask image (L mode is grayscale)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    # Draw a white rounded rectangle onto the black mask
    # The 'fill=255' creates the opaque area, the black background is transparent
    draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)

    # Apply the mask as the alpha channel of the original image
    image.putalpha(mask)

    # *** FIX 2: Must save output as PNG to maintain transparency ***
    output_path = "aligned_card_rounded.png"
    image.save(output_path)
    print(f"[INFO] Rounded card image saved to {output_path}")
    return output_path


# --- Main Execution Block ---

if __name__ == '__main__':
    # Set the path to your input image here
    # Example path assuming you ran the *first* Tesseract script once:
    input_image_path = 'orientation_corrected_image.jpeg'

    # Adjust this value (in pixels) to match your card's look
    corner_radius_pixels = 30

    # 1. Straighten the card using OpenCV
    aligned_image_path = process_card_image(input_image_path)

    if aligned_image_path:
        # 2. Round the corners using Pillow
        round_corners(aligned_image_path, radius=corner_radius_pixels)
