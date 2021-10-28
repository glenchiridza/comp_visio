import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("pose_videos/1.mp4")
prev_time = 0
while True:
    success, img = cap.read()

    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, f"{int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
