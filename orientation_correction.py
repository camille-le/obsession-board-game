import cv2
import pytesseract
from pytesseract import Output
import imutils

# Specify the path to the tesseract executable if it's not in your system's PATH
# Example for Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def correct_orientation(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return

    # Convert to RGB (Tesseract works better with RGB)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use Tesseract's OSD feature to detect orientation
    # psm 0 is for Orientation and Script Detection (OSD) only
    results = pytesseract.image_to_osd(rgb, config='--psm 0', output_type=Output.DICT)

    # Get the detected rotation angle required to correct the orientation
    rotate_angle = results["rotate"]
    print(f"[INFO] detected orientation: {results['orientation']} degrees")
    print(f"[INFO] rotate by {rotate_angle} degrees to correct")

    # If the image is upside down, rotate it 180 degrees
    if rotate_angle == 180:
        # Use imutils.rotate_bound to rotate without cropping corners
        rotated = imutils.rotate_bound(image, angle=rotate_angle)
        print("[INFO] Image was upside down, rotated 180 degrees.")
    else:
        # For other angles (0, 90, 270), OSD would correct it to 0, if needed.
        # If the goal is just to fix upside-down images, we keep the original.
        rotated = image
        print("[INFO] Image orientation is fine or needs a different correction than 180 degrees.")

    # Save or display the corrected image
    output_path = 'orientation_corrected_image.jpeg'
    cv2.imwrite(output_path, rotated)
    print(f"Corrected image saved to {output_path}")

    # Display images for verification (optional)
    cv2.imshow("Original", image)
    cv2.imshow("Output", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
# Make sure you have a test image file (e.g., 'scanned_card.jpg')
# correct_orientation('scanned_card.jpg')
correct_orientation('images/circle/1200 dpi/Scan 4.jpeg')