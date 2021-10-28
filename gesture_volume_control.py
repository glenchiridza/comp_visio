import cv2
import time
import numpy as np

# my modules import
import hand_tracking_module as htm

cam_width, cam_height = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)
prev_time = 0

detector = htm.handleDetector(detection_conf=0.7)

while True:
    success, img = cap.read()
    img = detector.locateHands(img)
    lm_list = detector.locatePosition(img, draw=False)
    if len(lm_list) != 0:
        # get tips of both thumb and index => 4and8 respectively
        print(lm_list[4], lm_list[8])

        x1,y1 = lm_list[4][1], lm_list[4][2]
        x2,y2 = lm_list[8][1], lm_list[8][2]

        cv2.circle(img, (x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x2,y2),10,(255,0,255),cv2.FILLED)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, f"fps{int(fps)}", (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow("ImageGC", img)
    cv2.waitKey(1)
