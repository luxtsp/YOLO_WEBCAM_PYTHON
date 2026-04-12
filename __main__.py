from ultralytics import YOLO
import sys

model = YOLO('yolo26n.pt')
model(int(sys.argv[1]), show=True, stream=True)

for result in results:
   boxes = result.boxes
   classes = result.names
