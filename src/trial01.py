import ultralytics
from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')

# Set the device to 'cpu'
device = 'cpu'
model.to(device)

# Training.
results = model.train(
    data='/Users/rianrachmanto/miniforge3/project/trial_ultrlytics/pothole_dataset_v8/pothole.yaml',
    imgsz=320,
    epochs=5,
    batch=16,
    project='yolov8n_custom',
    device=device
)



