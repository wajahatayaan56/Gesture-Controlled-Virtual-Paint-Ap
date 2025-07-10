import cv2
import mediapipe as mp
import pandas as pd
import time
import os
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Gestures & settings
gestures = ["draw", "select"]
samples_per_gesture = 100
output_csv = "gesture_data.csv"
image_output_dir = "gesture_images"

# Create folders
os.makedirs(image_output_dir, exist_ok=True)
for gesture in gestures:
    os.makedirs(os.path.join(image_output_dir, gesture), exist_ok=True)

# Utility
def draw_hand_preview(landmarks, preview_frame):
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * 200), int(lm.y * 200)
        cv2.circle(preview_frame, (cx, cy), 3, (0, 255, 0), -1)
    mp_draw.draw_landmarks(preview_frame, landmarks, mp_hands.HAND_CONNECTIONS)

def is_blurry(img, threshold=100.0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap < threshold

# Data holders
collected_data = []
labels = []

# Start camera
cap = cv2.VideoCapture(0)
print("\n📸 Camera started. Press ESC to cancel anytime.\n")

for gesture in gestures:
    sample_count = 0
    while sample_count < samples_per_gesture:
        # Countdown
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Get ready for '{gesture.upper()}' in {i}...",
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000)

        print(f"🔴 Recording '{gesture}'...")

        while sample_count < samples_per_gesture:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            preview_frame = np.zeros((200, 200, 3), dtype=np.uint8)

            valid_frame = False

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    draw_hand_preview(hand_landmarks, preview_frame)

                    row = []
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y])

                    # Check sharpness
                    if not is_blurry(frame):
                        collected_data.append(row)
                        labels.append(gesture)

                        # Save image
                        img_path = os.path.join(image_output_dir, gesture, f"{gesture}_{sample_count}.jpg")
                        cv2.imwrite(img_path, frame)
                        sample_count += 1
                        valid_frame = True

            # Info
            cv2.putText(frame, f"Recording: {gesture.upper()}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Samples: {sample_count}/{samples_per_gesture}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            frame[10:210, 430:630] = preview_frame
            if not valid_frame:
                cv2.putText(frame, "Skipping blurry or no-hand frame", (10, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print("Cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"Finished recording '{gesture}'.")

# Save CSV
df = pd.DataFrame(collected_data)
df["label"] = labels
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved {len(df)} samples to '{output_csv}'")
print(f"🧼 Blurry/no-hand frames were automatically skipped.\n")

cap.release()
cv2.destroyAllWindows()
