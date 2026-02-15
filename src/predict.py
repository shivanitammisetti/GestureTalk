import os
import joblib
import numpy as np

# -------- LOAD MODEL --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "models", "gesture_model.pkl")

model = joblib.load(model_path)

# -------- NORMALIZATION --------
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(21, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val != 0:
        landmarks = landmarks / max_val
    return landmarks.flatten().tolist()

# -------- PREDICTION FUNCTION --------
def predict_from_landmarks(raw_landmarks):
    norm = normalize_landmarks(raw_landmarks)
    norm = np.array(norm).reshape(1, -1)

    prediction = model.predict(norm)[0]

    if hasattr(model, "predict_proba"):
        conf = np.max(model.predict_proba(norm))
    else:
        conf = 1.0

    return prediction, conf

# import cv2
# import mediapipe as mp
# import joblib
# import numpy as np
# import pandas as pd
# from collections import deque
# from wordfreq import top_n_list
# import pyttsx3


# # ---------- WORD LIST ----------
# english_words = set(top_n_list("en", 50000))
# suggestions = []
# # --------------------------------

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

# # -------- WORD SUGGESTION FUNCTION (UPDATED: exact match first) --------
# def get_word_suggestions(current_word, limit=3):
#     if not current_word:
#         return []

#     current_word = current_word.lower()
#     matches = []

#     # exact match first
#     if current_word in english_words:
#         matches.append(current_word)

#     # prefix matches
#     prefix_matches = [
#         word for word in english_words
#         if word.startswith(current_word) and word != current_word
#     ]

#     matches.extend(prefix_matches)

#     return matches[:limit]
# # -----------------------------------------------------------------------

# # ---------------- SENTENCE BUFFER ----------------
# sentence = ""
# last_added = ""
# stable_letter = ""
# stable_count = 0
# STABLE_THRESHOLD = 8
# # -------------------------------------------------

# # ---------- TEXT TO SPEECH ----------
# def speak(text):
#     engine = pyttsx3.init()
#     engine.setProperty("rate", 150)
#     engine.say(text)
#     engine.runAndWait()
#     engine.stop()
# # -----------------------------------



# import os
# import joblib

# # Load trained model safely
# BASE_DIR = os.path.dirname(os.path.dirname(__file__))
# model_path = os.path.join(BASE_DIR, "models", "gesture_model.pkl")

# model = joblib.load(model_path)
# feature_names = model.feature_names_in_

# # Prediction smoothing buffer
# pred_buffer = deque(maxlen=15)

# # Camera
# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# mp_draw = mp.solutions.drawing_utils

# while True:
#     success, img = cap.read()
#     if not success:
#         break

#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)

#     display_text = ""
#     confidence = 0.0
#     show_warning = False

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

#             raw_landmarks = []
#             for lm in handLms.landmark:
#                 raw_landmarks.extend([lm.x, lm.y, lm.z])

#             if len(raw_landmarks) == 63:

#                 # -------- DISTANCE GUARD --------
#                 landmarks_np = np.array(raw_landmarks).reshape(21, 3)
#                 wrist = landmarks_np[0]
#                 hand_size = np.max(np.linalg.norm(landmarks_np - wrist, axis=1))

#                 if hand_size < 0.08:
#                     pred_buffer.clear()
#                     show_warning = True
#                     continue
#                 # --------------------------------

#                 norm_landmarks = normalize_landmarks(raw_landmarks)
#                 X = pd.DataFrame([norm_landmarks], columns=feature_names)

#                 pred = model.predict(X)[0]

#                 if hasattr(model, "predict_proba"):
#                     conf = np.max(model.predict_proba(X))
#                 else:
#                     conf = 1.0

#                 # Confidence gate
#                 if conf > 0.6:
#                     pred_buffer.append(pred)

#                 # Majority vote smoothing
#                 if len(pred_buffer) > 5:
#                     display_text = max(set(pred_buffer), key=pred_buffer.count)
#                     confidence = conf

#                     # -------- STABLE LETTER DETECTION --------
#                     if display_text == stable_letter:
#                         stable_count += 1
#                     else:
#                         stable_letter = display_text
#                         stable_count = 0

#                     if stable_count == STABLE_THRESHOLD:
#                         sentence += display_text
#                         last_added = display_text
#                         stable_count = 0

#                         # update suggestions
#                         current_word = sentence.split(" ")[-1]
#                         suggestions = get_word_suggestions(current_word)
#                     # -----------------------------------------

#     else:
#         pred_buffer.clear()

#     # ---------- USER FEEDBACK ----------
#     if show_warning:
#         cv2.putText(
#             img,
#             "Move hand closer",
#             (10, 90),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.9,
#             (0, 0, 255),
#             2
#         )

#     # Display prediction
#     cv2.putText(
#         img,
#         f"{display_text} ({confidence:.2f})",
#         (10, 50),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1.2,
#         (0, 255, 0),
#         3
#     )

#     # ---------- SHOW SENTENCE (UPDATED: scrolling display) ----------
#     display_sentence = sentence[-30:]  # show last 30 characters

#     cv2.putText(
#         img,
#         f"Sentence: {display_sentence}",
#         (10, 120),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (255, 0, 0),
#         2
#     )
#     # ---------------------------------------------------------------

#     # ---------- SHOW SUGGESTIONS ----------
#     y_pos = 160
#     for i, word in enumerate(suggestions):
#         cv2.putText(
#             img,
#             f"{i+1}: {word}",
#             (10, y_pos),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 255),
#             2
#         )
#         y_pos += 30
#     # -------------------------------------

#     cv2.imshow("Sign to Text - Live", img)

#     # -------- KEY CONTROLS --------
#     key = cv2.waitKey(1) & 0xFF

#     # ESC → exit
#     if key == 27:
#         break

#     # SPACE → add space
#     if key == ord(" "):
#         sentence += " "
#         last_added = ""
#         suggestions = []

#     # S → speak sentence
#     if key == ord("s"):
#         speak(sentence)


#     # BACKSPACE → delete last letter
#     if key == 8:
#         sentence = sentence[:-1]

#     # C → clear sentence
#     if key == ord("c"):
#         sentence = ""
#         last_added = ""
#         suggestions = []

#     # SELECT suggestion
#     if key in [ord("1"), ord("2"), ord("3")]:
#         index = int(chr(key)) - 1
#         if index < len(suggestions):
#             words = sentence.split(" ")
#             words[-1] = suggestions[index].upper()
#             sentence = " ".join(words) + " "
#             suggestions = []
#             last_added = ""
#     # ---------------------------------------

# cap.release()
# cv2.destroyAllWindows()

