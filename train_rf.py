import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ---------------------------------------
# STEP 1 — LOAD SEQUENCE DATASET
# ---------------------------------------
print("📥 loading sequence dataset...")

# no header, last column is label
df = pd.read_csv("final_fall_dataset.csv", header=None)

X = df.iloc[:, :-1]   # 3960 sequence features
y = df.iloc[:, -1]    # fall (1) / normal (0)

print("features shape:", X.shape)
print("labels shape:", y.shape)
print("\nlabel distribution:")
print(y.value_counts())

# ---------------------------------------
# STEP 2 — TRAIN / TEST SPLIT
# ---------------------------------------
print("\n📊 splitting train / test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# STEP 3 — RANDOM FOREST MODEL
# ---------------------------------------
print("\n🌲 training random forest on sequences...")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ---------------------------------------
# STEP 4 — EVALUATION
# ---------------------------------------
print("\n📈 evaluating model...")

y_pred = rf.predict(X_test)

print("\nconfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nclassification report:")
print(classification_report(y_test, y_pred, digits=4))

# ---------------------------------------
# STEP 5 — SAVE MODEL
# ---------------------------------------
joblib.dump(rf, "rf_fall_sequence_model.pkl")
print("\n💾 model saved as rf_fall_sequence_model.pkl")
print("🎉 sequence-based training complete!")
