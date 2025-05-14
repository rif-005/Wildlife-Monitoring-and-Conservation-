!pip install ultralytics opencv-python-headless
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load YOLOv5 model (pre-trained on COCO dataset)
model = YOLO("yolov5s.pt")

# Animal-related COCO classes
animal_labels = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'}
# Load the image
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)
fgmask = fgbg.apply(gray)

# Find contours (moving objects)
contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Set for storing detected animals
animal_detections = {}

# For each contour, detect and classify animals using YOLO
for contour in contours:
    if cv2.contourArea(contour) > 500:  # adjust sensitivity
        x, y, w, h = cv2.boundingRect(contour)

        # Crop the moving region of the image
        roi = image[y:y+h, x:x+w]
        if roi.size > 0:
            results = model.predict(source=roi, imgsz=320, conf=0.4, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    if label in animal_labels:
                        animal_detections[label] = animal_detections.get(label, 0) + 1
                        # Draw rectangle around the detected object
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Convert image for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show the resulting image with bounding boxes
plt.imshow(image_rgb)
plt.axis('off')
plt.title("Animal Detection in Image")
plt.show()
print("Detected animals in the image:")
for animal, count in animal_detections.items():
    print(f"- {animal}: {count}")
