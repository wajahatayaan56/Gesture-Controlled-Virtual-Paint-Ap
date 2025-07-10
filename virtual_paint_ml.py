import cv2
import numpy as np
import mediapipe as mp
import joblib
import math
import tensorflow as tf
import os

# Prompt for model selection
print("Select gesture recognition model:")
print("1. Random Forest (ML)")
print("2. CNN")
choice = input("Enter 1 or 2: ")

use_cnn = choice.strip() == '2'

if use_cnn:
    cnn_model = tf.keras.models.load_model("cnn_gesture_model.h5")
    cnn_classes = sorted(os.listdir("gesture_images"))
    # Adjust button positions and size for CNN (smaller and in top-right)
    cnn_buttons = [
        {"label": label, "x1": 520, "y1": 20 + i * 40, "x2": 600, "y2": 50 + i * 40}
        for i, label in enumerate(["Draw", "Erase", "Clear", "Red", "Green", "Blue"])
    ]
else:
    model = joblib.load('gesture_model.pkl')
    cnn_buttons = []

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Button definitions
buttons = [
    {"label": "Draw", "x1": 500, "y1": 20, "x2": 600, "y2": 60},
    {"label": "Erase", "x1": 500, "y1": 70, "x2": 600, "y2": 110},
    {"label": "Clear", "x1": 500, "y1": 120, "x2": 600, "y2": 160},
    {"label": "Red", "x1": 500, "y1": 170, "x2": 600, "y2": 210},
    {"label": "Green", "x1": 500, "y1": 220, "x2": 600, "y2": 260},
    {"label": "Blue", "x1": 500, "y1": 270, "x2": 600, "y2": 310},
]

# Initial values
draw_color = (255, 0, 255)
mode = "Draw"
xp, yp = 0, 0
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
selected_button = None
selection_cooldown = 0

# --------------------------
# ✋ Utility Functions
# --------------------------
def fingers_close(lm_list, id1, id2, threshold=30):
    x1, y1 = lm_list[id1]
    x2, y2 = lm_list[id2]
    return math.hypot(x2 - x1, y2 - y1) < threshold

def only_index_up(lm_list):
    fingers = []
    fingers.append(lm_list[4][0] < lm_list[3][0])  # Thumb
    fingers.append(lm_list[8][1] < lm_list[6][1])  # Index
    fingers.append(lm_list[12][1] < lm_list[10][1])  # Middle
    fingers.append(lm_list[16][1] < lm_list[14][1])  # Ring
    fingers.append(lm_list[20][1] < lm_list[18][1])  # Pinky
    return fingers[1] and not any(fingers[2:])

# --------------------------
# 🎨 Main Loop
# --------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if selection_cooldown > 0:
        selection_cooldown -= 1

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = [(int(lm.x * 640), int(lm.y * 480)) for lm in hand_landmarks.landmark]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            ix, iy = lm_list[8]

            # Gesture prediction
            if use_cnn:
                x1, y1 = max(ix - 64, 0), max(iy - 64, 0)
                x2, y2 = min(ix + 64, 640), min(iy + 64, 480)
                roi = frame[y1:y2, x1:x2]
                img = cv2.resize(roi, (128, 128))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)
                preds = cnn_model.predict(img)
                gesture = ["Draw", "Erase", "Clear", "Red", "Green", "Blue"][np.argmax(preds)]
            else:
                row = [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]
                gesture = model.predict([row])[0]

            # Show gesture
            cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Button interaction
            all_buttons = cnn_buttons if use_cnn else buttons
            for btn in all_buttons:
                if btn["x1"] < ix < btn["x2"] and btn["y1"] < iy < btn["y2"]:
                    cv2.rectangle(frame, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), (0, 255, 255), -1)
                    if fingers_close(lm_list, 8, 12) and selection_cooldown == 0:
                        selected_button = btn["label"]
                        selection_cooldown = 30
                        if selected_button.lower() == "clear":
                            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        elif selected_button.lower() == "red":
                            draw_color = (0, 0, 255)
                        elif selected_button.lower() == "green":
                            draw_color = (0, 255, 0)
                        elif selected_button.lower() == "blue":
                            draw_color = (255, 0, 0)
                        elif selected_button.lower() == "erase":
                            mode = "Erase"
                        elif selected_button.lower() == "draw":
                            mode = "Draw"

            # Drawing logic
            if only_index_up(lm_list):
                if mode == "Draw":
                    if xp == 0 and yp == 0:
                        xp, yp = ix, iy
                    cv2.line(canvas, (xp, yp), (ix, iy), draw_color, 5)
                    xp, yp = ix, iy
                elif mode == "Erase":
                    cv2.circle(canvas, (ix, iy), 20, (0, 0, 0), -1)
                    xp, yp = ix, iy
            else:
                xp, yp = 0, 0

    # Draw buttons
    all_buttons = cnn_buttons if use_cnn else buttons
    for btn in all_buttons:
        color = (200, 200, 200)
        if selected_button == btn["label"]:
            color = (0, 255, 0)
        cv2.rectangle(frame, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), color, -1)
        cv2.putText(frame, btn["label"], (btn["x1"] + 2, btn["y2"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Blend canvas
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Virtual Paint - CNN or ML", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
