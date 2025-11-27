import cv2
import numpy as np
from matplotlib import pyplot as plt

# --- CONFIGURATION ---
# NOTE: You MUST download the haarcascade XML file and update this path.
# It is typically found in your OpenCV install directory (e.g., in data/haarcascades/).
CASCADE_PATH = "training_data/haarcascade_frontalface_default.xml"
IMAGE_FILE = "scan1.jpeg"


def detect_face_and_crop_surrounding_circle(image_path, cascade_path):
    """
    1. Detects the face in the image to find its center.
    2. Uses the face's location to focus the search for the surrounding circle.
    3. Crops the original image based on the detected circle.
    """

    # --- 1. Load Image and Cascade ---
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Error: Image not loaded from {image_path}.")
        return None, None

    # Load the Haar Cascade for face detection
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except:
        print(f"Error: Could not load Cascade Classifier from {cascade_path}. Check path.")
        return None, original_img

    img_for_display = original_img.copy()
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    print("Image loaded and Face Cascade initialized.")

    # --- 2. Face Detection ---
    # Detect faces in the image. Faces is a list of (x, y, w, h) rectangles.
    # ScaleFactor and minNeighbors may need minor tweaking.
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        print("Error: No face detected. Cannot guide circle search.")
        return None, original_img

    # Assume the largest detected face is the main subject
    (fx, fy, fw, fh) = max(faces, key=lambda x: x[2] * x[3])

    # Calculate the center of the detected face (our expected circle center)
    face_center = (fx + fw // 2, fy + fh // 2)

    # Draw the face rectangle on the display image
    cv2.rectangle(img_for_display, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)
    cv2.circle(img_for_display, face_center, 5, (0, 0, 255), -1)
    print(f"Face detected at center {face_center}.")
    #

    # --- 3. Pre-process for Circle Detection ---
    # Smoothing is essential for clean Hough results
    smoothed_img = cv2.GaussianBlur(gray_img, (9, 9), 0)
    smoothed_img = cv2.medianBlur(smoothed_img, 7)  # Extra smoothing to clean up textures

    # --- 4. Hough Circle Transform Detection ---
    # We set parameters to look for a circle much larger than the face (min_radius)
    # and use the face's center to help filter the results later.
    circles = cv2.HoughCircles(
        smoothed_img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=40,  # Minimum distance between circle centers
        param1=50,
        param2=20,  # Lower sensitivity to catch faint decorative borders
        minRadius=int(fw * 1.5),  # Min radius must be larger than the face itself
        maxRadius=int(max(original_img.shape) // 2.5)  # Max radius set generously
    )

    # --- 5. Find the BEST Circle (Closest to Face Center) ---
    cropped_img = None

    if circles is not None:
        circles = np.uint16(np.around(circles))[0]

        # Filter: Find the circle whose center is closest to the detected face center
        def distance_to_face(circle):
            cx, cy, r = circle
            return np.sqrt((cx - face_center[0]) ** 2 + (cy - face_center[1]) ** 2)

        # Sort circles by their distance to the face center and pick the closest one
        best_circle = min(circles, key=distance_to_face)

        center_x, center_y, radius = best_circle
        padding = 30  # Extra padding for the final crop

        print(f"Best circle selected at center ({center_x}, {center_y}) with radius {radius}.")

        # --- 6. Calculate Bounding Box and Crop ---

        # Top-left corner
        x_start = max(0, center_x - radius - padding)
        y_start = max(0, center_y - radius - padding)

        # Bottom-right corner
        x_end = min(original_img.shape[1], center_x + radius + padding)
        y_end = min(original_img.shape[0], center_x + radius + padding)

        # Perform the crop on the ORIGINAL image
        cropped_img = original_img[y_start:y_end, x_start:x_end]

        # Draw the final chosen circle and bounding box
        cv2.circle(img_for_display, (center_x, center_y), radius, (0, 255, 0), 3)
        cv2.rectangle(img_for_display, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)

    else:
        print("No suitable circle detected near the face.")

    return cropped_img, img_for_display


# --- Main Execution ---

cropped_image_final, visualized_image_with_detection = detect_face_and_crop_surrounding_circle(
    IMAGE_FILE,
    CASCADE_PATH
)

if cropped_image_final is not None:
    cv2.imwrite('cropped_face_circle_guided.png', cropped_image_final)
    print("✅ Cropped image saved as 'cropped_face_circle_guided.png'.")

    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(cv2.cvtColor(visualized_image_with_detection, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Detection (Red Face, Green/Yellow Circle)')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(cropped_image_final, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Final Cropped Result')
    axes[1].axis('off')

    plt.show()
else:
    # If the process fails, show the image with the face detection (if any)
    if visualized_image_with_detection is not None:
        plt.imshow(cv2.cvtColor(visualized_image_with_detection, cv2.COLOR_BGR2RGB))
        plt.title('Detection Failed: Check Cascade Path or Hough Params')
        plt.axis('off')
        plt.show()