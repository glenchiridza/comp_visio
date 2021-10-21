import cv2
import mediapipe as mp
import time


class hand_detector():
    def __init__(self, mode=False, max_hands=2,
                 detection_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.max_hands, self.detection_conf,
            self.track_conf
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)

                # for idx, lm in enumerate(handlms.landmark):
                #     h, w, channel = img.shape
                #     cx, cy = int(lm.x * w), int(lm.y * h)
                #     print(idx, cx, cy)
                #     if idx == 0:
                #         cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


def main():
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 255, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()