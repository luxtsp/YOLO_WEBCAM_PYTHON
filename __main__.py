from ultralytics import YOLO
import sys
import cv2

model = YOLO('yolov8n.pt')  # or yolo11n.pt

cap = cv2.VideoCapture(int(sys.argv[1]))  # camera index, e.g. 0

# Create fullscreen window
cv2.namedWindow('YOLO Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('YOLO Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        classes = result.names
        # Draws boxes on the frame
        frame = result.plot()

    cv2.imshow('YOLO Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to quit
        break

cap.release()
cv2.destroyAllWindows()
