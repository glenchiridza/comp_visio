import cv2
import mediapipe as mp
import time


class PoseDetector:

    def __init__(self, mode=False, model_complx=1, smooth=True,
                 detect_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.model_complx = model_complx
        self.smooth = smooth
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complx,
                                     self.smooth, self.detect_conf, self.track_conf)

    def locatePose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self,img,draw=True):
        lm_list = []
        if self.results.pose_landmarks:
            for idx, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                print(lm.x, lm.y)
                cx, cy = int(lm.x * width), int(lm.y * height)
                lm_list.append([idx, cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 243, 255), cv2.FILLED)
        return lm_list

def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = PoseDetector()


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


if __name__ == "__main__":
    main()
