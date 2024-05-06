import cv2


if __name__ == '__main__':
    cap = cv2.VideoCapture("rtsp://192.168.144.25:8554/main.264")
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break