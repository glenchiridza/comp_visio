import cv2
import numpy as np

img = cv2.imread("res/runner.png")

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny = cv2.Canny(img,150,200)
imgDialation = cv2.dilate(imgCanny,)

cv2.imshow("Image Output",imgGray)
cv2.imshow("Image Blur",imgBlur)
cv2.imshow("Image Canny",imgCanny)

cv2.waitKey(0)

