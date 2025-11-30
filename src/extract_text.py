import cv2
import pytesseract
import numpy as np

# --- 1. CONFIGURATION ---

# ⚠️ IMPORTANT: Set the path to your Tesseract executable if it's not in your system's PATH.
# Example for Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the input image file path
IMAGE_PATH = 'output/aligned_card.png'


# Tesseract Configuration
# -l eng: Explicitly set English language
# --psm 6: Assume a single uniform block of text (often best for a card's main text area)
TESSERACT_CONFIG = r'-l eng --psm 6'


# --- 2. IMAGE PREPROCESSING & DEBUG FUNCTION ---

def preprocess_and_debug(img, roi):
    """
    Crops the image to the ROI, applies preprocessing, and displays the result
    for visual verification.
    """
    y_start, y_end, x_start, x_end = roi

    # 1. Crop the image to the defined text region
    cropped_img = img[y_start:y_end, x_start:x_end]

    if cropped_img.size == 0:
        raise ValueError("Cropping resulted in an empty image. Check the TEXT_ROI coordinates.")

    # 2. Convert to Grayscale and Apply Blur
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 3)

    # 3. Adaptive Thresholding (Best for uneven backgrounds)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,  # Block size
        10  # Constant subtracted
    )

    # --- DEBUGGING STEP: Display the image Tesseract will read ---
    cv2.imshow("OCR Input Check (Adjust TEXT_ROI)", thresh)
    print("📢 DEBUG: Displaying 'OCR Input Check' window. Press any key to continue...")
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()
    # -------------------------------------------------------------

    return thresh


# --- 3. MAIN EXTRACTION LOGIC ---

def extract_text_from_card():
    try:
        # Load the original image
        img = cv2.imread(IMAGE_PATH)

        if img is None:
            print(f"❌ ERROR: Image file not found at: {IMAGE_PATH}")
            return

        # 1. Preprocess the image and verify the crop
        processed_img = preprocess_and_debug(img, TEXT_ROI)

        # 2. Run Tesseract OCR
        extracted_text = pytesseract.image_to_string(processed_img, config=TESSERACT_CONFIG)

        print("\n✅ OCR Extraction Completed")
        print("---------------------------------")
        print(extracted_text.strip())
        print("---------------------------------")

    except pytesseract.TesseractNotFoundError:
        print("\n❌ ERROR: Tesseract not found. Check the installation and the path setting.")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Image file not found at: {IMAGE_PATH}")
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    extract_text_from_card()