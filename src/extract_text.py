import cv2
import pytesseract
import numpy as np
import os

# --- 1. CONFIGURATION ---

# ⚠️ IMPORTANT: Set the path to your Tesseract executable if it's not in your system's PATH.
# pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

# Define the input image file path
IMAGE_PATH = 'output/final_cards/Scan 3_rounded.png'
PROCESSED_IMAGE_PATH = 'output/temp.png'
CROPPED_1_PATH = 'output/temp_bottom_crop.png'
CROPPED_FINAL_PATH = 'output/temp_final_crop.png'

# Tesseract Configuration
TESSERACT_CONFIG = r'-l eng --psm 3'

# --- NEW: CROPPING PROPORTIONS ---
# How much of the image height to keep from the bottom (e.g., 1/3 = 0.3333)
BOTTOM_HEIGHT_RATIO = 1 / 3

# What percentage of the cropped image's width to keep in the center (e.g., 50)
CENTRAL_WIDTH_PERCENT = 50


# --- 2. IMAGE PREPROCESSING FUNCTION (No changes) ---

def preprocess_and_debug(img):
    """Applies the best-performing preprocessing parameters."""
    full_img = img
    gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 9)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        27,  # Block Size
        10  # Constant Subtracted
    )
    cv2.imwrite(PROCESSED_IMAGE_PATH, thresh)
    print(f"✅ Preprocessed image saved to: {PROCESSED_IMAGE_PATH}")
    return thresh


# --- 3. MAIN EXTRACTION LOGIC (Simplified Crop Logic) ---

def extract_text_from_card():
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract executable found and accessible.")

        img = cv2.imread(IMAGE_PATH)
        if img is None:
            print(f"❌ ERROR: Image file not found at: {IMAGE_PATH}")
            return

        print(f"✅ Image loaded successfully. Shape: {img.shape}")

        # 1. Preprocess the image
        processed_img = preprocess_and_debug(img)
        h, w = processed_img.shape[:2]

        # 2. CROP 1: Extract the **BOTTOM X%** (Height Crop)
        # Calculate start row based on the desired ratio (e.g., start at 1 - 1/3 = 2/3 of the height)
        start_row = int(h * (1.0 - BOTTOM_HEIGHT_RATIO))
        bottom_crop = processed_img[start_row:h, 0:w]

        print(f"✅ Step 1: Cropped to bottom {BOTTOM_HEIGHT_RATIO:.2%} (starting at row {start_row}). New shape: {bottom_crop.shape}")
        cv2.imwrite(CROPPED_1_PATH, bottom_crop)
        print(f"✅ Bottom crop image saved to: {CROPPED_1_PATH}")


        # 3. CROP 2: Extract the **CENTRAL Y% of the WIDTH**
        h_crop, w_crop = bottom_crop.shape[:2]

        # Calculate the ratio to be cut off from each side (e.g., (1 - 0.50) / 2 = 0.25)
        cut_ratio = (1.0 - (CENTRAL_WIDTH_PERCENT / 100.0)) / 2.0

        start_col = int(w_crop * cut_ratio)          # e.g., start at 25% mark
        end_col = int(w_crop * (1.0 - cut_ratio))    # e.g., end at 75% mark

        # Final Cropping: [rows, start_col:end_col]
        final_cropped_img = bottom_crop[0:h_crop, start_col:end_col]

        print(
            f"✅ Step 2: Cropped to central {CENTRAL_WIDTH_PERCENT}% width. Columns: {start_col} to {end_col}. Final shape: {final_cropped_img.shape}")

        # Save the final cropped image for verification
        cv2.imwrite(CROPPED_FINAL_PATH, final_cropped_img)
        print(f"✅ Final cropped image saved to: {CROPPED_FINAL_PATH}")

        # 4. Run Tesseract OCR on the **FINAL CROPPED IMAGE**
        print("📢 Starting Tesseract OCR extraction on final cropped area...")
        extracted_text = pytesseract.image_to_string(final_cropped_img, config=TESSERACT_CONFIG)

        # 5. Check and display result
        cleaned_text = extracted_text.strip()

        print("\n✅ OCR Extraction Completed")
        print("---------------------------------")
        if cleaned_text:
            print(f"**Extracted Text from Central {CENTRAL_WIDTH_PERCENT}% of Bottom {BOTTOM_HEIGHT_RATIO:.2%}:**\n{cleaned_text}")
        else:
            print("⚠️ **No text was extracted.** (Result was empty or whitespace only)")

        print("---------------------------------")

    except pytesseract.TesseractNotFoundError:
        print("\n❌ **CRITICAL ERROR: Tesseract not found.**")
        print("   -> The path you set for `pytesseract.tesseract_cmd` is incorrect.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    os.makedirs(os.path.dirname(PROCESSED_IMAGE_PATH), exist_ok=True)
    extract_text_from_card()