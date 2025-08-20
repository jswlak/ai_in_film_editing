from ultralytics import YOLO
import cv2
import os

# Load the model
model = YOLO("smoking_detection_trained2.pt")

# Paths
input_folder = "test_images"
output_folder = "output_images"
overlay_icon_path = "no_smoking.png"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the no smoking icon
icon = cv2.imread(overlay_icon_path, cv2.IMREAD_UNCHANGED)

# Resize icon if needed
scale_percent = 15  # adjust icon size
w = int(icon.shape[1] * scale_percent / 100)
h = int(icon.shape[0] * scale_percent / 100)
icon = cv2.resize(icon, (w, h))

# Loop through all images in the input folder
for img_name in os.listdir(input_folder):
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        # Run detection
        results = model(img_path)

        # Check if smoking is detected
        smoking_detected = False
        for r in results:
            if len(r.boxes) > 0:
                smoking_detected = True
                break

        # If detected, overlay the icon at bottom-left
        if smoking_detected:
            x_offset = 20  # small margin from left
            y_offset = img.shape[0] - h - 20  # margin from bottom

            if icon.shape[2] == 4:  # has alpha channel
                alpha_s = icon[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    img[y_offset:y_offset+h, x_offset:x_offset+w, c] = (
                        alpha_s * icon[:, :, c] +
                        alpha_l * img[y_offset:y_offset+h, x_offset:x_offset+w, c]
                    )
            else:
                img[y_offset:y_offset+h, x_offset:x_offset+w] = icon

        # Save final output
        output_path = os.path.join(output_folder, f"final_{img_name}")
        cv2.imwrite(output_path, img)
        print(f"✅ Processed {img_name} -> {output_path}")

print("✅ All images processed!")
