from ultralytics import YOLO

model = YOLO('yolo26n.pt')
model(0, show=True)

for result in results:
    boxes = result.boxes
    classes = result.names
