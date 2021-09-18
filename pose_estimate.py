import cv2
import mediapipe as mp

cap = cv2.VideoCapture("pose_videos/p1.avi")

while True:
    success, img = cap.read()
    cv2.imshow("Image",img)
    cv2.waitKey(1)

if __name__=="__main__":
    pass