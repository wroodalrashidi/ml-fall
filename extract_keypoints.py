import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# ---------------------------------------------------------
# INITIALIZE MEDIAPIPE POSE MODEL
# ---------------------------------------------------------
# mp.solutions.pose is Google's pre-trained skeleton detector.
# It finds 33 body landmarks (x, y, z).
mp_pose = mp.solutions.pose

# 'Pose' model is initialized once and reused for every frame.
pose = mp_pose.Pose(
    static_image_mode=False,   # This is a video, not images.
    model_complexity=1         # Medium accuracy/medium speed.
)

# This is where our final dataset of keypoints will be saved.
OUTPUT = "keypoints.csv"


# ---------------------------------------------------------
# EXTRACT KEYPOINTS FROM A SINGLE VIDEO
# ---------------------------------------------------------
# Input: video path + label (1 = fall, 0 = normal)
# Output: list of rows = [x1,y1,z1, x2,y2,z2, ... , label]
def extract_from_video(path, label):

    # Open video file
    cap = cv2.VideoCapture(path)
    rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Convert to RGB (mediapipe requires RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose detection
        res = pose.process(rgb)

        # If pose landmarks detected
        if res.pose_landmarks:

            lms = res.pose_landmarks.landmark  # list of 33 body points
            row = []

            # Extract each landmark: (x, y, z)
            for lm in lms:
                row += [lm.x, lm.y, lm.z]

            # Append label (fall or normal)
            row.append(label)

            # Add this frame’s keypoints as one data row
            rows.append(row)

    cap.release()
    return rows


# ---------------------------------------------------------
# EXTRACT KEYPOINTS FROM ALL VIDEOS IN A FOLDER
# ---------------------------------------------------------
# Loops through every .mp4 in a folder and extracts keypoints
def process_folder(folder, label):

    rows = []

    # Safety check — folder must exist
    if not os.path.exists(folder):
        print(f"⚠️ WARNING: Folder does NOT exist → {folder}")
        return rows

    # Walk through all files in the folder
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".mp4"):

                full = os.path.join(root, f)
                print("🎥 Extracting:", full)

                # Extract keypoints for this video
                rows.extend(extract_from_video(full, label))

    return rows


# ---------------------------------------------------------
# MAIN SCRIPT — RUNS WHEN YOU execute: python3 extract_keypoints.py
# ---------------------------------------------------------
if __name__ == "__main__":

    # Base dataset path
    BASE = "/Users/wrood/Desktop/machine-learning-fall-detection/dataset/public"

    # Folders that contain FALL videos → will be labeled as 1
    FALL_FOLDERS = [
        f"{BASE}/URFD/fall",
        f"{BASE}/GMDCSA/Subject 1/Fall",
        f"{BASE}/GMDCSA/Subject 2/Fall",
        f"{BASE}/GMDCSA/Subject 3/Fall",
        f"{BASE}/GMDCSA/Subject 4/Fall",
    ]

    # Folders that contain ADL/Normal videos → labeled as 0
    NORMAL_FOLDERS = [
        f"{BASE}/URFD/adl",
        f"{BASE}/GMDCSA/Subject 1/ADL",
        f"{BASE}/GMDCSA/Subject 2/ADL",
        f"{BASE}/GMDCSA/Subject 3/ADL",
        f"{BASE}/GMDCSA/Subject 4/ADL",
    ]

    all_rows = []

    # ---------------------------------------
    # Extract FALL dataset (label = 1)
    # ---------------------------------------
    print("\n📥 Starting FALL video extraction...")
    for folder in FALL_FOLDERS:
        all_rows.extend(process_folder(folder, 1))

    # ---------------------------------------
    # Extract NORMAL dataset (label = 0)
    # ---------------------------------------
    print("\n📥 Starting NORMAL (ADL) video extraction...")
    for folder in NORMAL_FOLDERS:
        all_rows.extend(process_folder(folder, 0))

    # ---------------------------------------
    # SAVE ALL KEYPOINTS INTO CSV
    # ---------------------------------------
    print(f"\n💾 Saving {len(all_rows)} samples to {OUTPUT}...")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    print("\n🎉 DONE! Keypoint extraction complete.\n")
