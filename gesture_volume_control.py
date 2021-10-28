import cv2
import time
import numpy as np

#my modules import
import hand_tracking_module as htm


cam_width, cam_height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)
prev_time = 0

detector = htm.handleDetector()


while True:
    success, img = cap.read()
    detector.locateHands(img)


    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, f"fps{int(fps)}", (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("ImageGC", img)
    cv2.waitKey(1)
