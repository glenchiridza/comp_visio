import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_conf=0.5):
        self.min_detection_conf = min_detection_conf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.face_detection = self.mpFaceDetection.FaceDetection(0.75)

    def find_faces(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(imgRGB)

        bounding_boxes = []

        if self.results.detections:
            for idx, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bounding_boxes.append([bbox, detection.score])
                self.fancy_draw(img,bbox)
                if draw:
                    cv2.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN,
                                3,
                                (255, 0, 255), 2)
        return img, bounding_boxes

    def fancy_draw(self, img, bbox, l=30, t=6):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), 1)
        # top left for x and y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # top right x1 and y1
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)

        # bottom right for x and y
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 + l), (255, 0, 255), t)
        # bottom right x1 and y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y - l), (255, 0, 255), t)

        return img

def main():
    cap = cv2.VideoCapture("pose_videos/1.mp4")
    pTime = 0
    detector = FaceDetector(0.75)

    while True:
        success, img = cap.read()
        img, bounding_boxes = detector.find_faces(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(20)


if __name__ == "__main__":
    main()
