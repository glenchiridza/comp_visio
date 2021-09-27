import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("pose_videos/1.mp4")
pTime = 0
while True:
    success, img = cap.read()
    cv2.imshow("Image", img)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}",(10,90),cv2.FONT_HERSHEY_PLAIN,)
    cv2.waitKey(1)
