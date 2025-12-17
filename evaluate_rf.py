import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------
# CONFIG
# ---------------------------------------
DATASET_PATH = "final_fall_dataset.csv"
MODEL_PATH = "rf_fall_sequence_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------
# LOAD DATASET
# ---------------------------------------
print("📥 loading dataset...")

df = pd.read_csv(DATASET_PATH, header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("samples:", X.shape[0])
print("features:", X.shape[1])

# ---------------------------------------
# TRAIN / TEST SPLIT (same logic as training)
# ---------------------------------------
print("\n📊 preparing test data...")

_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("test samples:", X_test.shape[0])

# ---------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------
print("\n📦 loading trained model...")
model = joblib.load(MODEL_PATH)

# ---------------------------------------
# SYSTEM EVALUATION
# ---------------------------------------
print("\n🧪 evaluating fall detection system...")

y_pred = model.predict(X_test)

# ---------------------------------------
# METRICS
# ---------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ accuracy:", round(accuracy, 4))

print("\n📌 confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 classification report:")
print(classification_report(y_test, y_pred, digits=4))

print("\n🎉 system evaluation complete")
