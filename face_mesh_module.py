import cv2
import mediapipe as mp
import time

class FaceMesh:
    def __init__(self,static_mode=False,max_faces=2,
                 min_detection_conf=0.5,
                 min_track_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_conf = min_detection_conf
        self.min_track_conf = min_track_conf


        mpDraw = mp.solutions.drawing_utils
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)

# change the size,  thickness of the circles or lines around it
draw_spec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(imgRGB)

    # display the results
    if results.multi_face_landmarks:
        for face_lm in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img,
                                  face_lm,
                                  mp_face_mesh.FACE_CONNECTIONS,
                                  draw_spec, draw_spec)
            # find all the different points
            for lm in face_lm.landmark:
                ih, iw, ic = img.shape
                x,y = int(lm.x * iw), int(lm.y * ih)




def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    while True:
        success, img = cap.read()
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time
        cv2.putText(img, f"Comp Scientist + Ecologist, everything will turn green", (20, 70), cv2.FONT_HERSHEY_PLAIN, 1.2,
                    (0, 255, 0), 2)

        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
