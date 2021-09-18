import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()




while True:
    success, img = cap.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hands only use RGB images
    results = hands.process(imageRGB)
    print(results.multi_hand_landmarks)
    # multi_hand_landmarks shows the coords if hand detected else None

    # extract multiple hands


    cv2.imshow("Image",img)
    cv2.waitKey(1)

