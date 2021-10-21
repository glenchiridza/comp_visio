import cv2
import time
import pose_estimate_module as pm

cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

while True:
    success, img = cap.read()
    img = detector.locatePose(img)
    lm_list = detector.getPosition(img)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(img, str(int(fps)), (60, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 243, 255, 255), 3)

    cv2.imshow("Image", img)

    cv2.waitKey(1)
