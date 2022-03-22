import cv2
import numpy as np


imgGray = cv2.imread('mapa_1.pgm')
fatia = imgGray[1400:2050, 1800:2600]
# print(imgGray[2000][2000])
# cv2.imshow("Original", fatia)
# cv2.waitKey(0)

linha1 = 100
y1 = 0
for i in range(len(fatia[linha1])):
    if fatia[linha1][i][0] == 0:
        y1 = i
        break;

print(y1)

linha2 = 630
y2 = 0
for i in range(len(fatia[linha2])):
    if fatia[linha2][i][0] == 0:
        y2 = i
        break;

print(y2)

print("y: ", y2-y1)
print("x: ", linha2-linha1)

theta = np.arctan2((linha2-linha1), (y2-y1))
print(np.rad2deg(theta))

cX = 164
cY = 645
M = cv2.getRotationMatrix2D((cX, cY), -theta, 1.0)

(h, w) = fatia.shape[:2]

rotated = cv2.warpAffine(fatia, M, (w, h))

#rotated = fatia

# cv2.imshow("Rotated by theta Degrees", rotated)
# cv2.waitKey(0)

treshold = np.zeros((h, w))

for i in range(len(rotated)):
    for j in range(len(rotated[i])):
        if rotated[i][j][0] < 206:
            treshold[i][j] = 0
        else:
            treshold[i][j] = 255

# cv2.imshow("Rotated by theta Degrees", treshold)
# cv2.waitKey(0)

# 164 645

# 156 87 , 714 85
# 162 645 , 718 646

final = treshold[86:645, 159:716]
cv2.imshow("Segmentada", final)
cv2.waitKey(0)
