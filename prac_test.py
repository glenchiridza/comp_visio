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
            # get the landmark information, x,y coords
            for idx, lm in enumerate(handlms.landmark):
                # get the pixels out of the image decimal values
                height,width,channels = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)
                print(idx,cx,cy)
                # if id of landmark is 0->hand bottom end, 4 if tip of thumb, and others fingers 1,2,3
                if idx == 0:
                    cv2.circle(img, (cx,cy),15,(255,234,255),cv2.FILLED)



            # use mediapipe function to draw all points on hand and there are almost 21 points
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 90), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
