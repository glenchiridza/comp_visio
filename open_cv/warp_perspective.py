import cv2
import numpy as np

img = cv2.imread("res/bc.jpg")
# ratio of card
width,height = 250,350
# get the values of the pixes of the specific image point you want to get using any graphics tool e.g paint
pts1 = np.float32([ [11,219], [287,188], [154,482] ,[352,440] ])
# define which is the first and which is thelast point, for the preceding defined points
pts2 = np.float32([ [0,0] , [width,0], [0,height], [width,height] ])
# transformation matrix required for the perspective itself
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow("Image",img)
cv2.imshow("WarpedImage",imgOutput)
cv2.waitKey(0)