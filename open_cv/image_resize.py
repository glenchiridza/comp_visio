import cv2

img = cv2.imread("res/msutop.jpg")
print(img.shape)

# cv2.imshow("School Top",img)

imgResize = cv2.resize(img, (500,700))  # on resize we have the width first argument and height as second argument
cv2.imshow("School Top",imgResize)
print(imgResize.shape)

#image cropping
imgCropped = img[0:200,240:450] # here we have the height as first argument and width as the second

cv2.imshow("Cropped Image",imgCropped)

cv2.waitKey(0)