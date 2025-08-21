from ultralytics import YOLO
import cv2
import os

# Load the model
model = YOLO("smoking_detection_trained2.pt")

# Paths
input_video = "test_videos/input.mp4"   # input video path
output_folder = "output_videos"
overlay_icon_path = "no_smoking.png"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load the no smoking icon
icon = cv2.imread(overlay_icon_path, cv2.IMREAD_UNCHANGED)

# Resize icon if needed
scale_percent = 15
w = int(icon.shape[1] * scale_percent / 100)
h = int(icon.shape[0] * scale_percent / 100)
icon = cv2.resize(icon, (w, h))

# Open video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("❌ Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Try multiple codecs automatically
codecs = [
    ('mp4v', 'final_output.mp4'),
    ('XVID', 'final_output.avi'),
    ('MJPG', 'final_output.avi')
]

out = None
output_path = None

for codec, filename in codecs:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    candidate_path = os.path.join(output_folder, filename)
    out = cv2.VideoWriter(candidate_path, fourcc, fps, (width, height))
    if out.isOpened():
        output_path = candidate_path
        print(f"✅ Using codec {codec}, saving to {output_path}")
        break
    else:
        print(f"⚠️ Codec {codec} failed, trying next...")

if out is None or not out.isOpened():
    print("❌ Error: Could not open any VideoWriter. Exiting.")
    cap.release()
    exit()

print("▶️ Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)
    smoking_detected = any(len(r.boxes) > 0 for r in results)

    # Overlay if detected
    if smoking_detected:
        x_offset = 20
        y_offset = frame.shape[0] - h - 20

        if icon.shape[2] == 4:  # alpha channel
            alpha_s = icon[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame[y_offset:y_offset+h, x_offset:x_offset+w, c] = (
                    alpha_s * icon[:, :, c] +
                    alpha_l * frame[y_offset:y_offset+h, x_offset:x_offset+w, c]
                )
        else:
            frame[y_offset:y_offset+h, x_offset:x_offset+w] = icon

    out.write(frame)

cap.release()
out.release()
print(f"🎉 Done! Saved processed video -> {output_path}")
