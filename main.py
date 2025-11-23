import numpy as np
import cv2

# Read in Image
img = cv2.imread('scan1.jpeg')

# Display Image in Window
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


