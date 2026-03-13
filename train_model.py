import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---------------------------
# (A) Load dataset from Excel
# ---------------------------
excel_path = "synthetic_seizure_dataset_20k.xlsx"  # <-- change if needed
df = pd.read_excel(excel_path, sheet_name="encoded")

expected_cols = ["HR", "HRV", "Medication", "Symptoms", "Sleep", "Stress", "risk"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in Excel sheet: {missing}")

X = df[["HR", "HRV", "Medication", "Symptoms", "Sleep", "Stress"]].astype(np.float32).values
y = df["risk"].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Loaded shape:", df.shape)
print("Risk distribution:\n", pd.Series(y).value_counts())

# ---------------------------
# (B) Train Random Forest (teacher)
# ---------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train, y_train)

rf_val_pred = rf.predict(X_val)
print("RF val accuracy:", accuracy_score(y_val, rf_val_pred))

# Soft labels from RF
y_train_soft = rf.predict_proba(X_train).astype(np.float32)  # (N,2)
y_val_soft   = rf.predict_proba(X_val).astype(np.float32)

# ---------------------------
# (C) Train small Keras model (student) to mimic RF
# ---------------------------
student = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # 6 inputs
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax"),
])

# ✅ Use the loss object (works across TF/Keras versions)
student.compile(
    optimizer="adam",
    loss=tf.keras.losses.KLDivergence(),
    metrics=["accuracy"]
)

student.fit(
    X_train, y_train_soft,
    validation_data=(X_val, y_val_soft),
    epochs=10,
    batch_size=256
)

# Optional: evaluate student vs true labels
student_val_pred = np.argmax(student.predict(X_val), axis=1)
print("Student val accuracy vs true labels:", accuracy_score(y_val, student_val_pred))

# ---------------------------
# (D) Convert student to TFLite
# ---------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(student)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional
tflite_model = converter.convert()

with open("seizure_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved: seizure_model.tflite")

# ---------------------------
# (E) Example: create input from app values
# ---------------------------
def encode_app_input(HR, HRV, Medication, Symptoms, Sleep, Stress):
    med = 1.0 if Medication.lower() == "yes" else 0.0
    sym = 1.0 if Symptoms.lower() == "yes" else 0.0
    slp = 1.0 if Sleep.lower() == "good" else 0.0
    strs = 1.0 if Stress.lower() == "high" else 0.0
    return np.array([[HR, HRV, med, sym, slp, strs]], dtype=np.float32)

sample = encode_app_input(92, 28, "no", "yes", "bad", "high")

pred_probs = student.predict(sample)
print("Prediction probs [low, high]:", pred_probs)
print("Predicted class:", int(np.argmax(pred_probs, axis=1)[0]))
