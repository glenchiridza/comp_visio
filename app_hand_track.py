import cv2
import time
import hand_tracking_module as htm


prevTime = 0
currTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handleDetector()

while True:
    success, img = cap.read()
    img = detector.locateHands(img)
    lmList = detector.locatePosition(img)
    if len(lmList) != 0:
        print(lmList[4])

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 90), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


