import cv2
import time
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('./yolov5', 'custom', path="train/best.pt", force_reload=True, source='local')
model.to(device)

if __name__ == '__main__':
    cap = cv2.VideoCapture("video/asibal.mp4")
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    delay = 60 # round(1000 / fps)
    out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
    print(f"delay:{delay}")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frame = np.array(frame)
            results = model(frame)

            for result in results.xyxy[0]:
                xmin, ymin, xmax, ymax, conf, cls = result

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 15, 255), 3)
                cv2.putText(frame, "tank", (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 좌표
                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                print(
                    f"Object class: {int(cls)}, Confidence: {conf:.2f}, Center coordinates: ({int(center_x)}, {int(center_y)})")

            cv2.imshow('Object Detection', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()