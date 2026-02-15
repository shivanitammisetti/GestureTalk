import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# ---------------- NORMALIZATION ----------------
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(21, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val != 0:
        landmarks = landmarks / max_val
    return landmarks.flatten().tolist()
# ------------------------------------------------


# -------- PATH SETUP (IMPORTANT FIX) --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")

os.makedirs(DATA_DIR, exist_ok=True)
# ---------------------------------------------


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Create CSV header if file doesn't exist
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

print("Press A–Z to save gesture | ESC to quit")

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    raw_landmarks = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
            for lm in handLms.landmark:
                raw_landmarks.extend([lm.x, lm.y, lm.z])

    cv2.imshow("Hand Tracking", img)

    key = cv2.waitKey(10)

    if key != -1:
        print("Key pressed:", key)

    # ESC to exit
    if key == 27:
        print("Exiting...")
        break

    # A-Z key press
    if (65 <= key <= 90) or (97 <= key <= 122):
        if len(raw_landmarks) == 63:
            label = chr(key).upper()
            norm_landmarks = normalize_landmarks(raw_landmarks)

            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([label] + norm_landmarks)

            print(f"Saved sample for label: {label}")

cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import csv
# import os
# import numpy as np

# # ---------------- NORMALIZATION ----------------
# def normalize_landmarks(landmarks):
#     landmarks = np.array(landmarks).reshape(21, 3)
#     wrist = landmarks[0]
#     landmarks = landmarks - wrist
#     max_val = np.max(np.linalg.norm(landmarks, axis=1))
#     if max_val != 0:
#         landmarks = landmarks / max_val
#     return landmarks.flatten().tolist()
# # ------------------------------------------------

# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# mp_draw = mp.solutions.drawing_utils

# DATA_DIR = "data"
# CSV_PATH = os.path.join(DATA_DIR, "landmarks.csv")
# os.makedirs(DATA_DIR, exist_ok=True)

# # Create CSV header
# if not os.path.exists(CSV_PATH):
#     with open(CSV_PATH, "w", newline="") as f:
#         writer = csv.writer(f)
#         header = ["label"]
#         for i in range(21):
#             header += [f"x{i}", f"y{i}", f"z{i}"]
#         writer.writerow(header)

# print("Press A–Z to save gesture | ESC to quit")

# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)

#     raw_landmarks = []

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)
#             for lm in handLms.landmark:
#                 raw_landmarks.extend([lm.x, lm.y, lm.z])

#     cv2.imshow("Hand Tracking", img)

#     key = cv2.waitKey(10)

#     if key != -1:
#         print("Key pressed:", key)

#     if key == 27:
#         print("Exiting...")
#         break

#     if (65 <= key <= 90) or (97 <= key <= 122):
#         if len(raw_landmarks) == 63:
#             label = chr(key).upper()
#             norm_landmarks = normalize_landmarks(raw_landmarks)

#             with open(CSV_PATH, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([label] + norm_landmarks)

#             print(f"Saved sample for label: {label}")

# cap.release()
# cv2.destroyAllWindows()

