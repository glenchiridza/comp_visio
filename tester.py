import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mPose = mp.solutions.pose
pose = mPose.Pose()

cap = cv2.VideoCapture(0)
prevTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    print(result.pose_landmarks)

    if result.pose_landmarks:
        mpDraw.draw_landmarks(img, result.pose_landmarks, mPose.POSE_CONNECTIONS)
        for idx, lm in enumerate(result.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 243, 255), cv2.FILLED)

    curr_time = time.time()
    fps = 1 / (curr_time - prevTime)
    prevTime = curr_time

    cv2.imshow("Image",img)
    cv2.waitKey(1)
