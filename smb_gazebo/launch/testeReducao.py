import numpy as np
import cv2

# l2 = 5
# l1 = 5
#
# costmap = np.zeros((l1, l2))
#
# costmap[2][2]=-1
#
# l2r = l2//4
# l1r = l1//4
#
# imagem = np.ones((l1r, l2r))*(-1)
#
# index_i = 0
# index_j = 0
#
# print(costmap)
#
# for i in range(0, l1, 4): # linha
#     for j in range(0, l2, 4): # coluna
#
#         obstaculo = False
#         for ii in range(i, i+4):
#             for jj in range(j, j+4):
#                 if costmap[jj][ii] == -1:
#                     print("entrei")
#                     obstaculo = True
#         if obstaculo:
#             imagem[i][j] = -1
#         else:
#             imagem[i][j] = 0
#         j = j+1
#     i = i+1

image = cv2.imread("map_lab2.png")

imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

half = cv2.resize(imgGray, (0, 0), fx = 0.5, fy = 0.5)

cv2.imshow("Reducao_Metade", half)
cv2.waitKey(0)
