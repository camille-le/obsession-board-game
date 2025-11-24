"""
Definitions:
* An HPF ("high pass filter") is a filter that examines a region of an image and boosts
the intensity of certain pixels based on the difference in intensity of the surrounding pixels.
* A kernel is a set of weights that are applied to a region in a source image to
generate a single pixel in the destination image.

We can think of a kernel as a piece of frosted glass moving over the source image
and letting a diffused blend of the source's light pass through.

A high-boost filter is a type of HPF and is effective in edge detection:
[0, -0.25, 0],
[-0.25, 1, -0.25],
[0, -0.25, 0]

Below are different techniques of applying the different filters.
"""
import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

# Define a 3x3 kernel size
kernel_3x3 = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

# Define a 5x5 kernel size
kernel_5x5 = np.array([[-1, -1, -1, -1, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, 2, 4, 2, -1],
                       [-1, 1, 2, 1, -1],
                       [-1, -1, -1, -1, -1]])

img = cv2.imread("gemini.png", cv2.IMREAD_GRAYSCALE)

# Convolution is a digital image processing technique
# ...that modifies an image by combining its pixels with
# ...a small, weighted matrix called a kernel (or filter).


# The first two methods are two HPFs with two different
# ...convolution kernels
k3_img = ndimage.convolve(img, kernel_3x3)
k5_img = ndimage.convolve(img, kernel_5x5)

# The third method is an HPF that we obtain by applying a LPF
# ...and calculating the difference between the original image
# ...a "differential" high-pass filter
blurred_img = cv2.GaussianBlur(img, (17, 17), 0)
hpf_img = img - blurred_img

# Define the data and titles
k3_img_rgb = cv2.cvtColor(k3_img, cv2.COLOR_BGR2RGB)
k5_img_rgb = cv2.cvtColor(k5_img, cv2.COLOR_BGR2RGB)

plots = [(k3_img_rgb, '3x3', None),
         (k5_img_rgb, '5x5', None),
         (blurred_img, 'Blurred', 'gray'),
         (hpf_img, "HPF", 'gray')]

# Create figure and subplots in one line (or very few)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Plot the images and set titles using iteration
for ax, (img, title, cmap) in zip(axes, plots):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()