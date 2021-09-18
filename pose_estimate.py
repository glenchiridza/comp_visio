import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture("pose_videos/1.mp4")
prev_time = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            print(lm.x, lm.y)
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(img, (cx, cy), 5, (255, 243, 255), cv2.FILLED)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (60, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 243, 255, 255), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(1)
