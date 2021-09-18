import cv2
import mediapipe as mp
import time


prevTime = 0
currTime = 0

class handleDetector():
    def __init__(self, mode=False,max_hands =2,
                 detection_conf=0.5,track_conf=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_conf = detection_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_hands,
                                        self.detection_conf,self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def locateHands(self, img, draw=True):
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # hands only use RGB images
        results = self.hands.process(imageRGB)
        print(results.multi_hand_landmarks)
        # multi_hand_landmarks shows the coords if hand detected else None

        # extract multiple hands
        if results.multi_hand_landmarks:
            for handlms in results.multi_hand_landmarks:
                if draw:
                    # use mediapipe function to draw all points on hand and there are almost 21 points
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img



# # get the landmark information, x,y coords
#                 for idx, lm in enumerate(handlms.landmark):
#                     # get the pixels out of the image decimal values
#                     height, width, channels = img.shape
#                     cx, cy = int(lm.x * width), int(lm.y * height)
#                     print(idx, cx, cy)
#                     # if id of landmark is 0->hand bottom end, 4 if tip of thumb, and others fingers 1,2,3
#                     if idx == 0:
#                         cv2.circle(img, (cx, cy), 15, (255, 234, 255), cv2.FILLED)


def main():
    prevTime = 0
    currTime = 0
    cap = cv2.VideoCapture(0)
    detector = handleDetector()

    while True:
        success, img = cap.read()
        img = detector.locateHands(img)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10, 90), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
