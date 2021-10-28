import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("pose_videos/1.mp4")
prev_time = 0

mpDraw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)

# change the size,  thickness of the circles or lines around it
draw_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=10)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    # display the results
    if results.multi_face_landmarks:
        for face_lm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,
                                  face_lm,
                                  mp_face_mesh.FACE_CONNECTIONS,
                                  draw_spec, draw_spec)

    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, f"{int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", img)
    cv2.waitKey(1)
