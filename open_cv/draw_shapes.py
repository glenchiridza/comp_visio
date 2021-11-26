import cv2
import numpy as np


img = np.zeros((512,512,3),np.uint8)

# color the complete image
# img[:] = 255,0,0
# color a specific range
#img[100:200,200:500] = 255,7,0

# create line
cv2.line(img,(0,0),(300,300),(0,255,0),3)
# stretch line end to end
cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(255,0,255),1)

# rectangle
cv2.rectangle(img,(0,0),(240,350),(255,0,255),2)
# fill the rectangle
cv2.rectangle(img,(250,360),(400,450),(0,0,255),cv2.FILLED)

# circles
cv2.circle(img,(40,50),30,(255,255,0),5)

# put text
cv2.putText(img,"Helolo this is glen yoh",(150,100),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,125,0),1)

cv2.imshow("Image",img)
print(img)

cv2.waitKey(0)