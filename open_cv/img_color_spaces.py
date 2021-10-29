import cv2

img = cv2.imread("res/me.jpg")

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)

# edge detector, canny edge detector
imgCanny = cv2.Canny(img, 100, 100)

cv2.imshow("GrayScale Image", imgGray)
cv2.imshow("Blur Image", imgBlur)
cv2.waitKey(0)
