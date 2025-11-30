from PIL import Image
import pytesseract

# Set the path to the Tesseract executable (if not in your system's PATH)

# Path to your image file
image_path = '../input/circle/1200 dpi/Scan 4.jpeg'

try:
    # Open the image using Pillow
    img = Image.open(image_path)

    # Use pytesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(img)

    # Print the extracted text
    print("Extracted Text:")
    print(extracted_text)

except pytesseract.TesseractNotFoundError:
    print("Tesseract is not installed or not in your PATH. Please install Tesseract OCR.")
except FileNotFoundError:
    print(f"Image file not found at: {image_path}")
except Exception as e:
    print(f"An error occurred: {e}")