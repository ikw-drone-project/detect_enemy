# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import torch
import pymysql
import numpy as np
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', path="train/best.pt", force_reload=True)
model.to(device)

def detect_objects_image(video = "video\DJI_0003_.MP4"):
    # cap = cv2.VideoCapture(video_source)
    cap = cv2.imread(video,cv2.IMREAD_COLOR)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Object Detection', frame)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"width : {width}\nheight : {height}\nfps : {fps}\n")
    #
    #
    # results = model(cap)
    #
    # for result in results.xyxy[0]:
    #     xmin, ymin, xmax, ymax, conf, cls = result
    #     if conf >= 0.6:
    #         cv2.rectangle(cap, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 15, 255), 3)
    #         cv2.putText(cap, "tank", (int(xmin), int(ymin) - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #
    #         # 좌표
    #         center_x = (xmin + xmax) / 2
    #         center_y = (ymin + ymax) / 2
    #         print(
    #             f"Object class: {int(cls)}, Confidence: {conf:.2f}, Center coordinates: ({int(center_x)}, {int(center_y)})")

    cv2.destroyAllWindows()

def detect_objects(video_source=0):
    cap = cv2.VideoCapture(video_source)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"width : {width}\nheight : {height}\nfps : {fps}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        frame = np.array(frame)

        if not ret:
            break

        results = model(frame)

        for result in results.xyxy[0]:
            xmin, ymin, xmax, ymax, conf, cls = result
            if conf >= 0.6:
                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 15, 255), 3)
                cv2.putText(frame, "tank", (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 좌표
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                print(
                    f"Object class: {int(cls)}, Confidence: {conf:.2f}, Center coordinates: ({int(center_x)}, {int(center_y)})")

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(device)
    # detect_objects_image()
    detect_objects()
