import cv2

img = cv2.imread("res/me.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("GrayScale Image", imgGray)

cv2.waitKey(0)