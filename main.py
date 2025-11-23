import cv2

# Read in Image
img = cv2.imread('scan1.jpeg')

# Get Gray Image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Get Blurred Image
kernel_size = (11, 11)
blurred_img = cv2.GaussianBlur(gray_img, kernel_size, 0)

# Show Main Image
cv2.imshow("Image", img)
cv2.waitKey(0)

# Show Gray Image
cv2.imshow("Gray Image", gray_img)
cv2.waitKey(0)

# Show BLurred Image
cv2.imshow("Blurred Image", blurred_img)
cv2.waitKey(0)

# Clean up state
cv2.destroyAllWindows()

#
# ## Canny Edge Detection
#
# # from matplotlib import pyplot as plt
# #
# # edges = cv2.Canny(img, 100, 200)
# #
# # plt.subplot(121), plt.imshow(gray_img, cmap = 'gray')
# # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# # plt.subplot(122), plt.imshow(edges, cmap = 'gray')
# # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# # plt.show()
# #
