import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
model.classes = [1, 2, 3, 5]  # bicycle, car, motorcycle, bus

cap = cv2.VideoCapture('video_input/traffic_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated = results.render()[0]
    cv2.imshow('YOLOv5 Vehicle Detection', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

