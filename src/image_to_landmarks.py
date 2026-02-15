import cv2
import mediapipe as mp
import os
import csv
import shutil
from datetime import datetime

DATASET_DIR = "data"  # now using your existing data folder
CSV_PATH = "data/landmarks.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Step 1️⃣: Backup existing CSV
if os.path.exists(CSV_PATH):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{CSV_PATH.replace('.csv','')}_backup_{timestamp}.csv"
    shutil.copy(CSV_PATH, backup_path)
    print(f"Backup created: {backup_path}")

# Step 2️⃣: Create CSV if it doesn't exist
if not os.path.exists(CSV_PATH):
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

# Step 3️⃣: Process images
count = 0

for img_name in os.listdir(DATASET_DIR):
    img_path = os.path.join(DATASET_DIR, img_name)

    # Skip if not an image
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    # Assume label is first character of filename (e.g., A_img1.jpg)
    label = img_name[0].upper()

    img = cv2.imread(img_path)
    if img is None:
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            row = [label]

            for lm in handLms.landmark:
                row.extend([lm.x, lm.y, lm.z])

            if len(row) == 64:
                with open(CSV_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                    count += 1

print(f"\nDONE ✅ Total new samples added: {count}")

