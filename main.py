# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import time
import torch
import gps_to_db
import numpy as np
import sys

# -=-=-=-=-=-=-=-= Params -=-=-=-=-=-=-=-= #
lat = 36.168421
lng = 128.467292
alt = 100 # meter
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('./yolov5', 'custom', path="train/best.pt", force_reload=True, source='local')
model.to(device)

timer = 0
event = False
last_detected = False
def detect_objects(video_source=0):
    global last_detected
    global timer

    cap = cv2.VideoCapture(video_source)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"width : {width}\nheight : {height}\nfps : {fps}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No Signal")
            time.sleep(1)
        
        else:
            frame = np.array(frame)
            results = model(frame)

            for result in results.xyxy[0]:
                xmin, ymin, xmax, ymax, conf, cls = result

                # FIXME:
                #   정확도... 80%이상 일치하는 경우
                if conf >= 0.8:
                    if timer == 0:
                        timer = time.time()
                    elif (time.time() - timer > 2):
                        print("80% tank detected!")
                        gps_to_db.save_current(lat, lng)
                        timer = 0

                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 15, 255), 3)
                    cv2.putText(frame, "tank", (int(xmin), int(ymin) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 좌표
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    print(f"탱크 감지:, {conf*100:.0f}%, coordinates: ({int(center_x)}, {int(center_y)})")

                else :
                    last_detected = False
            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(device)
    # FIXME:
    #   웹 표현 알고리즘 문제 有
    #   적 위치가 아닌 현재 위치에 기반하여 지도상에 표기...
    # gps_to_db.save_enemies(lat,lng)
    detect_objects() #"rtsp://192.168.144.25:8554/main.264"
