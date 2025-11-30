import cv2
import numpy as np
import imutils
import os
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
INPUT_DIR = 'input/circle/1200 dpi'  # Path to the folder containing your scans
OUTPUT_DIR = 'output/final_cards'  # Folder where final images will be saved
CORNER_RADIUS_PIXELS = 30  # Radius for rounded corners


# --- UTILITY FUNCTIONS ---

def order_points(pts):
    """
    Orders a list of 4 points in top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def four_point_transform(image, pts):
    """
    Applies perspective transform to a region defined by 4 points to get a top-down view.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


# --- MODIFIED PROCESSING FUNCTIONS ---

def process_card_image(image_path, temp_output_path):
    """
    Uses OpenCV to find the card edges and straighten the perspective.
    Saves the aligned image to a temporary PNG file.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image {image_path}")
        return None

    # Card edge detection logic (unchanged)
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
        print(f"[ERROR] Could not find a 4-point contour in {os.path.basename(image_path)}. Skipping.")
        return None

    warped = four_point_transform(image, screenCnt.reshape(4, 2))

    # Save to a temporary path as PNG to maintain quality
    cv2.imwrite(temp_output_path, warped)
    return temp_output_path


def round_corners(image_path, radius, final_output_path):
    """
    Applies rounded, transparent corners and saves the final image as PNG.
    This resolves the 'cannot write mode RGBA as JPEG' error.
    """
    # Image must support RGBA
    image = Image.open(image_path).convert("RGBA")

    # Create a mask image
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([(0, 0), image.size], radius=radius, fill=255)

    # Apply the mask
    image.putalpha(mask)

    # Save the final image as PNG, which supports transparency (RGBA)
    image.save(final_output_path)
    return final_output_path


# --- Main Execution Block ---

if __name__ == '__main__':
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Starting processing for files in: {INPUT_DIR}")

    # Loop through all files in the input directory
    for filename in os.listdir(INPUT_DIR):
        # Check for image extensions and ensure it's a file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')) and os.path.isfile(
                os.path.join(INPUT_DIR, filename)):

            print(f"\nProcessing: {filename}...")

            # Define all paths
            input_file_path = os.path.join(INPUT_DIR, filename)

            # Temporary path for the aligned image (must be PNG for quality/next step)
            temp_aligned_path = os.path.join(OUTPUT_DIR, f"temp_aligned_{filename}.png")

            # 💡 FIX: Define the final output path to ALWAYS use the .png extension.
            # This is essential because the corner rounding creates transparency (RGBA mode),
            # which can only be saved to formats like PNG.
            name_part, ext_part = os.path.splitext(filename)
            final_output_path = os.path.join(OUTPUT_DIR, f"{name_part}_rounded.png")  # Force .png extension

            aligned_path = None

            # --- STAGE 1: Straighten the card ---
            try:
                aligned_path = process_card_image(input_file_path, temp_aligned_path)
            except Exception as e:
                print(f"   ❌ ERROR in straightening {filename}: {e}")
                continue

            # --- STAGE 2: Round the corners ---
            if aligned_path and os.path.exists(aligned_path):
                try:
                    round_corners(aligned_path, CORNER_RADIUS_PIXELS, final_output_path)
                    print(f"   -> Final rounded card saved to: {final_output_path}")

                except Exception as e:
                    print(f"   ❌ ERROR in rounding corners for {filename}: {e}")

                finally:
                    # Clean up the temporary aligned file
                    if os.path.exists(temp_aligned_path):
                        os.remove(temp_aligned_path)

            else:
                print(f"   ⚠️ Warning: Alignment failed or path not found for {filename}.")

    print("\nProcessing complete! All final files saved as PNG in the output directory.")
