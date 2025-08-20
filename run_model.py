from ultralytics import YOLO
import cv2

# Load the model
model = YOLO("smoking_detection_trained2.pt")

# Run inference on image
results = model("test_image3.png")

# Load the no smoking icon
icon = cv2.imread("no_smoking.png", cv2.IMREAD_UNCHANGED)

# Resize icon if needed
scale_percent = 15  # adjust icon size
w = int(icon.shape[1] * scale_percent / 100)
h = int(icon.shape[0] * scale_percent / 100)
icon = cv2.resize(icon, (w, h))

# Load original image
img = cv2.imread("test_image3.png")

# Loop through detections
for r in results:
    if len(r.boxes) == 0:
        continue  # skip if no smoking detected

    # Overlay on bottom-left corner
    x_offset = 20
    y_offset = img.shape[0] - h - 20

    if icon.shape[2] == 4:  # PNG with alpha channel
        alpha_s = icon[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            img[y_offset:y_offset+h, x_offset:x_offset+w, c] = (
                alpha_s * icon[:, :, c] + alpha_l * img[y_offset:y_offset+h, x_offset:x_offset+w, c]
            )
    else:
        img[y_offset:y_offset+h, x_offset:x_offset+w] = icon

# Save final output
cv2.imwrite("final_output3.png", img)
print("✅ Detection complete. Check final_output.jpg")
