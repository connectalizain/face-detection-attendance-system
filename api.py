from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import pickle
import os
from webhook_logger import log_attendance_webhook
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load encodings
ENCODINGS_PATH = "known_faces/encodings.pkl"
known_encodings = []
known_metadata = []

def load_encodings():
    global known_encodings, known_metadata
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
            known_encodings = data["encodings"]
            known_metadata = data["metadata"]
        print(f"Loaded {len(known_encodings)} encodings")
    else:
        print("No encodings file found!")

load_encodings()

@app.route("/recognize", methods=["POST"])
def recognize():
    if "frame" not in request.files:
        return jsonify({"error": "No frame uploaded"}), 400

    file = request.files["frame"]
    img_array = np.frombuffer(file.read(), np.uint8)
    import cv2
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.55)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        if True in matches:
            best_idx = int(np.argmin(distances))
            metadata = known_metadata[best_idx]
            log_attendance_webhook(
                name=metadata["name"],
                guardian_no=metadata.get("guardian_no", ""),
                guardian_name=metadata.get("guardian_name", ""),
                school_name=metadata.get("school_name", "")
            )
            results.append({"name": metadata["name"], "status": "recognized"})
        else:
            results.append({"name": "Unknown", "status": "unknown"})

    return jsonify({"faces": results, "count": len(results)})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "encodings_loaded": len(known_encodings)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)