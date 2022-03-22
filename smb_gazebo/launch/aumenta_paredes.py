# Import packages
import cv2
import numpy as np
# import imutils

img = cv2.imread('Segmentada.png')
print(img.shape) # Print image shape
#fatia = img[1400:2050, 1800:2600]
# rotated = imutils.rotate(fatia,-2)
# rotated_Cropped = rotated[81:637, 147:710]
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
(thresh, bwIm) = cv2.threshold(grayImage, 253, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((36, 36), 'uint8')
dilate_img = cv2.dilate(bwIm, kernel, iterations=1)
(thresh, bwIm2) = cv2.threshold(dilate_img, 253, 255, cv2.THRESH_BINARY_INV)

# for i in range(len(grayImage)):
#     for j in range(len(grayImage[i])):
#         if grayImage[i][j]==0:
#             bwIm2[i][j] = 150

cv2.imshow('Dilated Image 2', bwIm2)
cv2.waitKey(0)


cv2.imwrite("map.png", bwIm2)
cv2.waitKey(0)
