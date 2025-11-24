import cv2
import numpy as np

# Create an empty black image
img = np.zeros((200, 200), dtype=np.uint8)

# Place empty white square in the center
img[50:150, 50:150] = 255

# Threshold the image
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
# Draw green contours on the image
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img = cv2.drawContours(color, contours, -1, (0,255,0), 2)

# Display the image
cv2.imshow("contours", color)
cv2.waitKey()
cv2.destroyAllWindows()