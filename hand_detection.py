import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hands only use RGB images
    results = hands.process(imageRGB)
    print(results.multi_hand_landmarks)
    # multi_hand_landmarks shows the coords if hand detected else None

    # extract multiple hands
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            # use mediapipe function to draw all points on hand and there are almost 21 points
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 90), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
