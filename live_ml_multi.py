
# import cv2
# import time
# import csv
# import os
# import numpy as np
# import joblib
# from datetime import datetime
# from ultralytics import YOLO
# from twilio.rest import Client
# import mediapipe as mp

# # =====================================================================
# # LOGGING SYSTEM
# # =====================================================================

# LOG_FILE = "fall_events_log.csv"

# # Create CSV file if not exists
# if not os.path.exists(LOG_FILE):
#     with open(LOG_FILE, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "event_id", "person_id", "date", "time",
#             "location", "camera_source", "status"
#         ])

# def log_fall_event(person_id, location="Main Camera", camera_source="iPhone Camera"):
#     """Logs a confirmed fall into a CSV file IMMEDIATELY (no buffering)."""
#     now = datetime.now()
#     date_str = now.strftime("%Y-%m-%d")
#     time_str = now.strftime("%H:%M:%S")

#     # Determine next event ID
#     with open(LOG_FILE, "r") as f:
#         event_id = sum(1 for _ in f)

#     with open(LOG_FILE, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             event_id, person_id, date_str, time_str,
#             location, camera_source, "FALL CONFIRMED"
#         ])

#         f.flush()         # 🔥 write immediately
#         os.fsync(f.fileno())  # 🔥 force save to disk

#     print(f"[LOG] Fall event #{event_id} recorded for Person {person_id}")



# # =====================================================================
# # TWILIO WHATSAPP CONFIG
# # =====================================================================

# ACCOUNT_SID = "AC439514ffd5e7015a93e8dca8331733bf"
# AUTH_TOKEN  = "2500da1a01446751e6bc25c152657308"

# FROM_WHATSAPP = "whatsapp:+14155238886"
# TO_WHATSAPP   = "whatsapp:+96595589155"

# twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# def send_fall_alert(person_id):
#     """Sends WhatsApp alert."""
#     try:
#         twilio_client.messages.create(
#             body=f"FALL ALERT: Person {person_id} has fallen and stayed motionless for {CONFIRM_FALL_SEC} seconds.",
#             from_=FROM_WHATSAPP,
#             to=TO_WHATSAPP
#         )
#         print(f"[TWILIO] WhatsApp alert sent for Person {person_id}")
#     except Exception as e:
#         print("[TWILIO] Error sending WhatsApp:", e)


# # =====================================================================
# # LOAD FALL MODEL + MEDIAPIPE
# # =====================================================================

# model = joblib.load("fall_model.pkl")

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(
#     static_image_mode=False,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# def extract_keypoints(landmarks):
#     row = []
#     for lm in landmarks:
#         row.extend([lm.x, lm.y, lm.z])
#     return np.array(row).reshape(1, -1)


# # =====================================================================
# # MULTI-PERSON FALL DETECTION STATE
# # =====================================================================

# fall_start_times = {}      # Start time of detected fall
# alert_sent_flags = {}      # Prevent multiple WhatsApp alerts
# logged_flags = {}          # Prevent multiple CSV logs

# CONFIRM_FALL_SEC = 5


# # =====================================================================
# # YOLO PERSON DETECTOR
# # =====================================================================

# yolo = YOLO("yolov8n.pt")


# # =====================================================================
# # CAMERA (webcam or iPhone)
# # =====================================================================

# # Webcam:
# cap = cv2.VideoCapture(0)

# # iPhone:
# # cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

# print("ResQ Multi-Person Fall Detection Running... Press ESC to exit.")


# # =====================================================================
# # MAIN LOOP
# # =====================================================================

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = yolo(frame, conf=0.5)

#     for r in results:
#         for person_id, box in enumerate(r.boxes):

#             # Only detect persons
#             if int(box.cls[0]) != 0:
#                 continue

#             x1, y1, x2, y2 = map(int, box.xyxy[0])

#             person_crop = frame[y1:y2, x1:x2]
#             rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
#             res = pose.process(rgb_crop)

#             state = "NO PERSON"
#             color = (0, 255, 255)

#             if res.pose_landmarks:
#                 lm = res.pose_landmarks.landmark

#                 # Visibility check: 70% of landmarks must be visible
#                 visible_count = sum(p.visibility > 0.5 for p in lm)
#                 if visible_count < 25:
#                     state = "BODY NOT FULLY IN FRAME"
#                     color = (0, 255, 255)
#                     fall_start_times.pop(person_id, None)
#                     alert_sent_flags.pop(person_id, None)
#                     logged_flags.pop(person_id, None)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     continue

#                 # Shoulders + hips must be visible
#                 vital_keypoints = [11, 12, 23, 24]
#                 if not all(lm[j].visibility > 0.5 for j in vital_keypoints):
#                     state = "BODY NOT FULLY IN FRAME"
#                     color = (0, 255, 255)
#                     fall_start_times.pop(person_id, None)
#                     alert_sent_flags.pop(person_id, None)
#                     logged_flags.pop(person_id, None)
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                     continue

#                 # Predict fall
#                 keypoints = extract_keypoints(lm)
#                 pred = model.predict(keypoints)[0]

#                 if pred == 1:
#                     if person_id not in fall_start_times:
#                         fall_start_times[person_id] = time.time()

#                     elapsed = time.time() - fall_start_times[person_id]

#                     if elapsed >= CONFIRM_FALL_SEC:
#                         state = "FALL CONFIRMED"
#                         color = (0, 0, 255)

#                         # LOG ONLY ONCE
#                         if not logged_flags.get(person_id, False):
#                             log_fall_event(person_id)
#                             logged_flags[person_id] = True

#                         # ALERT ONLY ONCE
#                         if not alert_sent_flags.get(person_id, False):
#                             send_fall_alert(person_id)
#                             alert_sent_flags[person_id] = True

#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

#                     else:
#                         state = f"FALL SUSPECTED ({int(elapsed)}s)"
#                         color = (0, 140, 255)

#                 else:
#                     # Reset person state when standing up
#                     state = "OK"
#                     color = (0, 255, 0)
#                     fall_start_times.pop(person_id, None)
#                     alert_sent_flags.pop(person_id, None)
#                     logged_flags.pop(person_id, None)

#             # Draw status
#             cv2.putText(frame, f"Person {person_id}: {state}",
#                         (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#     cv2.imshow("ResQ Multi-Person Fall Detection", frame)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import time
import csv
import os
import numpy as np
import joblib
from datetime import datetime
from ultralytics import YOLO
from twilio.rest import Client
import mediapipe as mp

# =====================================================================
# CONFIG
# =====================================================================

SEQUENCE_LENGTH = 40
CONFIRM_FALL_SEC = 5
LOG_FILE = "fall_events_log.csv"
YOLO_CONF = 0.4              # lower = better recall
VISIBILITY_THRESH = 15       # relaxed visibility
MISSING_TIMEOUT = 2.0        # seconds before cleanup

# =====================================================================
# LOGGING SYSTEM
# =====================================================================

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id", "person_id", "date", "time",
            "location", "camera_source", "status"
        ])

def log_fall_event(person_id, location="Main Camera", camera_source="Live Camera"):
    now = datetime.now()
    with open(LOG_FILE, "r") as f:
        event_id = sum(1 for _ in f)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            event_id,
            person_id,
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            location,
            camera_source,
            "FALL CONFIRMED"
        ])
        f.flush()
        os.fsync(f.fileno())

    print(f"[LOG] Fall event #{event_id} recorded for Person {person_id}")

# =====================================================================
# TWILIO WHATSAPP CONFIG
# =====================================================================

ACCOUNT_SID = "AC439514ffd5e7015a93e8dca8331733bf"
AUTH_TOKEN  = "2500da1a01446751e6bc25c152657308"

FROM_WHATSAPP = "whatsapp:+14155238886"
TO_WHATSAPP   = "whatsapp:+96595589155"

twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_fall_alert(person_id):
    try:
        twilio_client.messages.create(
            body=f"🚨 FALL ALERT: Person {person_id} has fallen and remained motionless for {CONFIRM_FALL_SEC} seconds.",
            from_=FROM_WHATSAPP,
            to=TO_WHATSAPP
        )
        print(f"[TWILIO] Alert sent for Person {person_id}")
    except Exception as e:
        print("[TWILIO] Error:", e)

# =====================================================================
# LOAD MODEL + MEDIAPIPE
# =====================================================================

model = joblib.load("rf_fall_sequence_model.pkl")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_keypoints(landmarks):
    row = []
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row

# =====================================================================
# STATE TRACKING
# =====================================================================

person_buffers = {}
fall_start_times = {}
alert_sent_flags = {}
logged_flags = {}
last_seen = {}

# =====================================================================
# YOLO PERSON DETECTOR
# =====================================================================

yolo = YOLO("yolov8n.pt")

# =====================================================================
# CAMERA
# =====================================================================

cap = cv2.VideoCapture(0)
print("ResQ Live Fall Detection Running... Press ESC to exit.")

# =====================================================================
# MAIN LOOP
# =====================================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, conf=YOLO_CONF)

    current_time = time.time()

    for r in results:
        for box in r.boxes:

            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🔑 stable pseudo person ID
            person_id = int(x1 / 50) * 1000 + int(y1 / 50)

            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb_crop)

            state = "LOW CONFIDENCE"
            color = (0, 255, 255)

            last_seen[person_id] = current_time

            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                visible = sum(p.visibility > 0.5 for p in lm)

                if visible >= VISIBILITY_THRESH:
                    keypoints = extract_keypoints(lm)

                    person_buffers.setdefault(person_id, []).append(keypoints)
                    if len(person_buffers[person_id]) > SEQUENCE_LENGTH:
                        person_buffers[person_id].pop(0)

                    if len(person_buffers[person_id]) == SEQUENCE_LENGTH:
                        seq = np.array(person_buffers[person_id]).flatten()

                        if seq.shape[0] == 3960:
                            pred = model.predict(seq.reshape(1, -1))[0]

                            if pred == 1:
                                fall_start_times.setdefault(person_id, current_time)
                                elapsed = current_time - fall_start_times[person_id]

                                if elapsed >= CONFIRM_FALL_SEC:
                                    state = "FALL CONFIRMED"
                                    color = (0, 0, 255)

                                    if not logged_flags.get(person_id, False):
                                        log_fall_event(person_id)
                                        logged_flags[person_id] = True

                                    if not alert_sent_flags.get(person_id, False):
                                        send_fall_alert(person_id)
                                        alert_sent_flags[person_id] = True
                                else:
                                    state = f"FALL SUSPECTED ({int(elapsed)}s)"
                                    color = (0, 140, 255)
                            else:
                                state = "OK"
                                color = (0, 255, 0)
                                fall_start_times.pop(person_id, None)
                                alert_sent_flags.pop(person_id, None)
                                logged_flags.pop(person_id, None)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"Person {person_id}: {state}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # 🧹 cleanup missing persons
    for pid in list(last_seen.keys()):
        if current_time - last_seen[pid] > MISSING_TIMEOUT:
            person_buffers.pop(pid, None)
            fall_start_times.pop(pid, None)
            alert_sent_flags.pop(pid, None)
            logged_flags.pop(pid, None)
            last_seen.pop(pid, None)

    cv2.imshow("ResQ Multi-Person Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
