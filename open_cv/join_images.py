import cv2
import numpy as np

# note both of images should have the same channels, if gray scale both should be gray scale, if colored both should be colored
# cant resize

img = cv2.imread("res/bc.jpg")

imgHor = np.hstack((img,img))
imgVer  = np.vstack((img,img))

cv2.imshow("Horizontal Stacked Image",imgHor)
cv2.imshow("Vertical Stacked Image",imgVer)

cv2.waitKey(0)