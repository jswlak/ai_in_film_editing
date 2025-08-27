from ultralytics import YOLO

# Load your pretrained model
model = YOLO("smoking_detection_trained2.pt")

# Print model summary
model.info()
