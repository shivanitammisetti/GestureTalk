import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------- PATH SETUP --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(BASE_DIR, "data", "landmarks.csv")
model_path = os.path.join(BASE_DIR, "models", "gesture_model.pkl")

# -------- LOAD DATASET --------
df = pd.read_csv(data_path)

# Features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------- SAVE MODEL --------
joblib.dump(model, model_path)
print("\nModel saved in models/gesture_model.pkl")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# import joblib

# # Load dataset
# df = pd.read_csv("data/landmarks.csv")

# # Features and labels
# X = df.drop("label", axis=1)
# y = df["label"]

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Train model
# model = RandomForestClassifier(
#     n_estimators=300,
#     max_depth=20,
#     random_state=42,
#     n_jobs=-1
# )

# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)

# print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred))

# # Save model
# joblib.dump(model, "gesture_model.pkl")
# print("\nModel saved as gesture_model.pkl")

