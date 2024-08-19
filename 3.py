import time
import cv2 as cv
import utils
import mediapipe as mp
import numpy as np
import tensorflow as tf  # TFLite model requires TensorFlow

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection

def classify_face(face_roi):
    # Preprocess the face ROI for the TFLite model (resize, normalize, etc.)
    input_shape = input_details[0]['shape']
    face_roi_resized = cv.resize(face_roi, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(face_roi_resized, axis=0)
    input_data = (input_data / 255.0).astype(np.float32)  # Normalization if needed

    # Set the tensor to point to the input data to be inferred.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the interpreter
    interpreter.invoke()

    # Get the result from the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data), output_data  # Returns class index and confidence

cap = cv.VideoCapture(0)
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
) as face_detector:
    frame_counter = 0
    fonts = cv.FONT_HERSHEY_PLAIN
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = cap.read()
        if ret is False:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        results = face_detector.process(rgb_frame)
        frame_height, frame_width, c = frame.shape
        if results.detections:
            for face in results.detections:
                face_react = np.multiply(
                    [
                        face.location_data.relative_bounding_box.xmin,
                        face.location_data.relative_bounding_box.ymin,
                        face.location_data.relative_bounding_box.width,
                        face.location_data.relative_bounding_box.height,
                    ],
                    [frame_width, frame_height, frame_width, frame_height],
                ).astype(int)
                x, y, w, h = face_react
                padding = 50
                fx_min = max(x - padding, 0)
                fx_max = min(x + padding + w, frame_width)
                fy_min = max(y - padding, 0)
                fy_max = min(y + padding + h, frame_height)

                face_roi = frame[fy_min:fy_max, fx_min:fx_max]

                # Classify the face ROI
                class_index, _ = classify_face(face_roi)

                # Blur the face if classified as 'Female'
                if class_index == 2:  # Assuming class 1 is 'Female'
                    face_blur_roi = cv.blur(face_roi, (53, 53))
                    frame[fy_min:fy_max, fx_min:fx_max] = face_blur_roi

                cv.imshow("face_roi", face_roi)
                

                utils.rect_corners(frame, face_react, utils.MAGENTA, th=3)
                utils.text_with_background(
                    frame,
                    f"score: {(face.score[0]*100):.2f}",
                    face_react[:2],
                    fonts,
                    color=utils.MAGENTA,
                    scaling=0.7,
                )
        fps = frame_counter / (time.time() - start_time)
        utils.text_with_background(frame, f"FPS: {fps:.2f}", (30, 30), fonts)
        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
