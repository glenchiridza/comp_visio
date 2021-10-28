import cv2
import mediapipe as mp
import time


class FaceMeshDetector:
    def __init__(self, static_mode=False, max_faces=2,
                 min_detection_conf=0.5,
                 min_track_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_conf = min_detection_conf
        self.min_track_conf = min_track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.static_mode, self.max_faces,
                                                    self.min_detection_conf, self.min_track_conf)

        # change the size,  thickness of the circles or lines around it
        self.draw_spec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def find_face_mesh(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(imgRGB)

        # display the results
        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                faces = []
                if draw:
                    self.mpDraw.draw_landmarks(img,
                                               face_lm,
                                               self.mp_face_mesh.FACE_CONNECTIONS,
                                               self.draw_spec, self.draw_spec)
                # find all the different points
                face = []
                for lm in face_lm.landmark:
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # single face landmarks
                    face.append([x,y])
            # append the landmarks of all the faces
            faces.append(face)
        return img, faces

def main():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    detector = FaceMeshDetector
    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img)
        if len(faces) != 0:
            print(faces)
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time)
        prev_time = cur_time
        cv2.putText(img, f"{int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    (0, 255, 0), 2)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
