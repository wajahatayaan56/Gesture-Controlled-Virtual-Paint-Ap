import cv2
import numpy as np
import tensorflow as tf
import os

# Load model and class labels
model = tf.keras.models.load_model("cnn_gesture_model.h5")
class_labels = sorted(os.listdir("gesture_images"))
img_size = 128

# Webcam
cap = cv2.VideoCapture(0)

print("📷 Starting webcam. Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show rectangle where to place hand
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Preprocess ROI for model
    img = cv2.resize(roi, (img_size, img_size))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    pred_label = class_labels[np.argmax(preds)]

    # Display prediction
    cv2.putText(frame, f"Gesture: {pred_label}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Live Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
