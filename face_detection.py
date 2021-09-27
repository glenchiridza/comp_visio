import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("pose_videos/1.mp4")
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
face_detection = mpFaceDetection.FaceDetection()


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = face_detection.process(imgRGB)

    if results.detections:
        for idx, detection in enumerate(results.detections):
            print(idx, detection)
            print(detection.score)
            print(detection.location_data.relative_bounding_box)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(20)
