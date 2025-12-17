import os
import cv2
import csv
import numpy as np
import mediapipe as mp

# ============================================================
# CONFIGURATION
# ============================================================

BASE = "../dataset/public"
SEQ_LEN = 30
STRIDE = 5
OUTPUT_CSV = "keypoints_seq_velocity_all.csv"

FALL = 1
ADL = 0

# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_pose(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if not res.pose_landmarks:
        return None

    keypoints = []
    for lm in res.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y])

    return keypoints


def compute_velocity(seq):
    seq = np.array(seq)
    vel = np.diff(seq, axis=0)
    vel = np.vstack([vel, vel[-1]])
    return vel.flatten()


def make_sequences(frames, label):
    rows = []
    for i in range(0, len(frames) - SEQ_LEN + 1, STRIDE):
        seq = frames[i:i + SEQ_LEN]
        row = np.concatenate([
            np.array(seq).flatten(),
            compute_velocity(seq),
            [label]
        ])
        rows.append(row.tolist())
    return rows


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        kp = extract_pose(frame)
        if kp is not None:
            frames.append(kp)

    cap.release()

    if len(frames) < SEQ_LEN:
        return []

    return make_sequences(frames, label)


def process_video_with_frame_range(video_path, start_f, end_f, label):
    """
    NEW (CORRECT) LE2i processing:
    only frames inside annotated fall window
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 1  # LE2i annotations are 1-indexed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if start_f <= frame_idx <= end_f:
            kp = extract_pose(frame)
            if kp is not None:
                frames.append(kp)

        frame_idx += 1

    cap.release()

    if len(frames) < SEQ_LEN:
        return []

    return make_sequences(frames, label)


# ============================================================
# IMAGE SEQUENCE PROCESSING (URFD)
# ============================================================

def process_image_sequence(folder, label):
    frames = []
    images = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".png")
    ])

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        kp = extract_pose(img)
        if kp is not None:
            frames.append(kp)

    if len(frames) < SEQ_LEN:
        return []

    return make_sequences(frames, label)


# ============================================================
# DATASET PROCESSORS
# ============================================================

def process_urfd_adl():
    rows = []
    base = os.path.join(BASE, "URFD", "adl")

    for root, dirs, _ in os.walk(base):
        for d in dirs:
            rows.extend(
                process_image_sequence(os.path.join(root, d), ADL)
            )

    return rows


def process_gmdcsa():
    rows = []
    base = os.path.join(BASE, "GMDCSA")

    for subject in os.listdir(base):
        subj_path = os.path.join(base, subject)
        if not os.path.isdir(subj_path):
            continue

        for label_name, label in [("Fall", FALL), ("ADL", ADL)]:
            folder = os.path.join(subj_path, label_name)
            if not os.path.exists(folder):
                continue

            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith(".mp4"):
                        rows.extend(
                            process_video(os.path.join(root, f), label)
                        )

    return rows


# ============================================================
# ❌ OLD LE2i CODE (WRONG – KEPT FOR DOCUMENTATION)
# ============================================================
#
# def process_le2i():
#     """
#     OLD METHOD:
#     labels LE2i videos using video index
#     ignores annotation files
#     includes walking frames before fall
#     """
#     def le2i_label(video_name):
#         num = int(video_name.split("(")[1].split(")")[0])
#         return ADL if num <= 15 else FALL
#
#     ...


# ============================================================
# ✅ NEW LE2i CODE (ANNOTATION-BASED)
# ============================================================

def read_le2i_annotation(txt_path):
    fall_start = None
    fall_end = None

    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.isdigit():
                if fall_start is None:
                    fall_start = int(line)
                elif fall_end is None:
                    fall_end = int(line)
                    break

    if fall_start is None or fall_end is None:
        raise ValueError(f"Invalid annotation: {txt_path}")

    return fall_start, fall_end


def process_le2i():
    rows = []
    base = os.path.join(BASE, "Le2i")

    for scene in os.listdir(base):
        scene_path = os.path.join(base, scene)
        if not os.path.isdir(scene_path):
            continue

        videos_dir = os.path.join(scene_path, scene, "Videos")
        ann_dir = os.path.join(scene_path, scene, "Annotation_files")

        if not os.path.exists(videos_dir) or not os.path.exists(ann_dir):
            continue

        for f in os.listdir(videos_dir):
            if not f.lower().endswith(".avi"):
                continue

            txt_path = os.path.join(ann_dir, f.replace(".avi", ".txt"))
            if not os.path.exists(txt_path):
                continue

            fall_start, fall_end = read_le2i_annotation(txt_path)

            rows.extend(
                process_video_with_frame_range(
                    os.path.join(videos_dir, f),
                    fall_start,
                    fall_end,
                    FALL
                )
            )

    return rows


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    all_rows = []

    all_rows.extend(process_urfd_adl())
    all_rows.extend(process_gmdcsa())
    all_rows.extend(process_le2i())

    print(f"\n💾 saving {len(all_rows)} samples")

    with open(OUTPUT_CSV, "w", newline="") as f:
        csv.writer(f).writerows(all_rows)

    print("🎉 DONE")
