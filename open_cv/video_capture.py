import cv2

vid = cv2.VideoCapture(0)
# vid size 3 = width, 4 = height
vid.set(3, 640)
vid.set(4, 480)
# vid brightness id = 10
vid.set(10, 100)

while True:
    success, img = vid.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
