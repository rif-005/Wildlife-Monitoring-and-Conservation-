# Wildlife Monitoring and Conservation 

## Purpose

The primary purpose of this code is to automate wildlife monitoring and conservation efforts by detecting and classifying animals in uploaded images. This approach aims to streamline the identification process, helping researchers track animal populations, understand behavioral patterns, and assess the impact of environmental changes — ultimately supporting conservation efforts.



## Technologies Used

- **Ultralytics YOLOv5** – A state-of-the-art deep learning model for real-time object detection, pre-trained on the COCO dataset (includes animal classes).
- **OpenCV** – For image processing tasks like background subtraction and contour detection.
- **NumPy** – For efficient numerical computations and array handling.
- **Matplotlib** – For visualizing processed images with detected animals.



## Usage

1. **Environment Setup**
   - Install required libraries:
     ```bash
     pip install ultralytics opencv-python-headless
     ```

2. **Image Upload**
   - Upload the image containing animals via the file upload interface (e.g., Google Colab or Jupyter).

3. **Image Processing**
   - Convert the image to grayscale.
   - Apply background subtraction to detect moving or distinct objects (potential animals).

4. **Animal Detection**
   - Use YOLOv5 to classify detected contours as known animal types based on the COCO dataset.
   - Draw bounding boxes around detected animals for visual feedback.

5. **Results Display**
   - Print the detected animal types and their counts.
   - Display the final image with bounding boxes.



## Conclusion

This code offers a practical solution for wildlife researchers and conservationists by providing a fast, automated method for monitoring animal populations. By leveraging deep learning through YOLOv5 and efficient image processing via OpenCV, the project significantly reduces manual labor, increases accuracy, and improves the speed of species identification in ecological studies.
