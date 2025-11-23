"""
An HPF ("high pass filter") is a filter that examines a region of an image and boosts
the intensity of certain pixels based on the difference in intensity of the surrounding pixels.

A kernel is a set of weights that are applied to a region in a source image to
generate a single pixel in the destination image.

We can think of a kernel as a piece of frosted glass moving over the source image
and letting a diffused blend of the source's light pass through.

A high-boost filter is a type of HPF and is effective in edge detection:
[0, -0.25, 0],
[-0.25, 1, -0.25],
[0, -0.25, 0]

Below is an example of applying an HPF to an image.
"""
import cv2
import numpy as np
from scipy import ndimage

kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv2.imread("scan1.jpeg", cv2.IMREAD_GRAYSCALE)
k3 = ndimage.convolve(img, kernel_3x3)
k5 = ndimage.convolve(img, kernel_5x5)

blurred = cv2.GaussianBlur(img, (17, 17), 0)
g_hpf = img - blurred

cv2.imshow("3x3", k3)
cv2.waitKey()

cv2.imshow("5x5", k5)
cv2.waitKey()

cv2.imshow("blurred", blurred)
cv2.waitKey()

cv2.imshow("g_hpf", g_hpf)
cv2.waitKey()
cv2.destroyAllWindows()
