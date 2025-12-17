from flask import Flask, render_template, jsonify, Response
import csv
from datetime import datetime
import pytz
import cv2

app = Flask(__name__)

LOG_FILE = "fall_events_log.csv"

# ================= READ + NORMALIZE EVENTS =================
def read_events():
    events = []
    kuwait = pytz.timezone("Asia/Kuwait")

    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # ---- combine date + time ----
            try:
                dt = datetime.strptime(
                    f"{row['date']} {row['time']}",
                    "%Y-%m-%d %H:%M:%S"
                )
                dt = kuwait.localize(dt)
                timestamp = dt.isoformat()
            except Exception:
                timestamp = ""

            # ---- normalize for frontend ----
            event = {
                "id": row.get("event_id", "—"),
                "image": row.get("camera_source", "—"),
                "fall": "True" if "FALL" in row.get("status", "") else "False",
                "timestamp": timestamp
            }

            events.append(event)

    # newest first
    return events[::-1]

# ================= ROUTES =================
@app.route("/")
def dashboard():
    return render_template("index.html")

@app.route("/events")
def events():
    return jsonify(read_events())

# ================= LIVE CAMERA =================
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
