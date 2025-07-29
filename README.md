# PersonBlurusingMediapipe

## Overview

This project leverages computer vision techniques, specifically MediaPipe and TensorFlow Lite, to detect faces in real-time video streams and apply selective blurring based on classification results. It also utilizes a custom-trained model to distinguish between different classes (such as gender) and applies privacy-preserving blurring to selected individuals (e.g., females).

## Features

- **Real-Time Face Detection**: Uses MediaPipe for highly accurate and efficient face detection in live webcam feeds.
- **Face Classification**: Integrates a TensorFlow Lite model to classify detected faces into categories (e.g., male or female).
- **Selective Blurring**: Automatically blurs the face region of a person if classified into a specific category (e.g., female), while leaving others unaltered.
- **Custom Visualizations**: Employs utility functions for drawing bounding boxes, corners, and overlaying text for score and FPS metrics.

## Code Structure

- `3.py`: The main script for running real-time face detection and blurring.
  - Loads a TFLite classification model.
  - Captures video from a webcam.
  - Detects faces using MediaPipe.
  - For each detected face:
    - Extracts the region of interest (ROI).
    - Classifies the face using the TFLite model.
    - If the face belongs to the targeted class (e.g., female), applies a strong blur to the ROI.
    - Draws rectangles, overlays classification confidence, and displays FPS.
  - Example snippet:
    ```python
    if class_index == 2:  # Assuming class 2 is 'Female'
        face_blur_roi = cv.blur(face_roi, (53, 53))
        frame[fy_min:fy_max, fx_min:fx_max] = face_blur_roi
    ```

- `utils.py`: Contains helper functions for:
  - Drawing rectangles and stylized corners on faces.
  - Displaying text with backgrounds.
  - Drawing transparent polygons and circles for overlays.
  - Calculating and displaying FPS metrics.

- `model0.ipynb`: A Jupyter notebook demonstrating the model training process.
  - Uses Roboflow to download and prepare datasets.
  - Trains an object detection or classification model with MediaPipe Model Maker.
  - Converts the model to TensorFlow Lite format for use in the main application.

## How Person Blurring Works

1. **Detection**: Each frame from the webcam is processed with MediaPipe to detect faces.
2. **Classification**: Each face is classified using the provided TFLite model.
3. **Blurring**: If the face matches the blurring criteria, a Gaussian blur is applied to the region.
4. **Visualization**: Bounding boxes, class scores, and FPS are overlayed for transparency and debugging.

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- MediaPipe
- TensorFlow (for TFLite model inference)
- Roboflow (for dataset management and model training)
- Jupyter Notebook (for training and experimentation)

## License

MIT License (see LICENSE file).

## Review

- **Strengths**:
  - Modular design with clear separation between detection, classification, and visualization logic.
  - Real-time performance with FPS tracking.
  - Use of open-source libraries and reproducible training workflow.
- **Considerations**:
  - The class index for blurring is hard-coded; parameterization could improve flexibility.
  - The model's accuracy and fairness rely on the quality and balance of the training data.
  - Privacy: Be mindful of ethical implications of face detection and classification.

## Usage

To run the real-time blurring script:

```bash
python 3.py
```

Ensure the TFLite model is available in the project directory as `model.tflite`.

To retrain or experiment with the model, open `model0.ipynb` in Google Colab or Jupyter Notebook.

---

For more details, refer to the code and model training notebook.
