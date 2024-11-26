import cv2
import numpy as np
from object_detection import ObjectDetection



od = ObjectDetection()
cap = cv2.VideoCapture("videos/video.mp4")

while True:
    _, frame = cap.read()
    (class_ids, scores, boxes) = od.detect(frame)

    for box in boxes:
        # print(box)
        (x,y,w,h) = box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
        

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key==27:
        break
cap.release()
cv2.destroyAllWindows()


