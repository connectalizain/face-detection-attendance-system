# Core face recognition logic

import os
import pickle
import sys

import cv2
import face_recognition
import numpy as np


ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "encodings.pkl")
MATCH_THRESHOLD = 0.5

# --- Performance tuning ---
# How often to run face recognition (every N frames).
# Higher = faster playback, less CPU. Start at 5, go up to 8-10 if still slow.
FRAME_SKIP = 3

# Scale factor for detection. 0.25 is much faster than 0.5 on old hardware.
DETECTION_SCALE = 0.5

# --- Stability tuning ---
# How many consecutive detections needed before showing/announcing a name.
CONFIRM_HITS = 2

# How many frames to keep showing a confirmed name after detection stops.
HOLD_FRAMES = 20


def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        print(f"Error: encodings file not found at {ENCODINGS_PATH}. Run bulk_enroll.py first.")
        sys.exit(1)

    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)

    if not data:
        print("Error: encodings file is empty. Enroll some students first.")
        sys.exit(1)

    known_encodings = [entry["encoding"] for entry in data]
    known_names = [entry["name"] for entry in data]
    return known_encodings, known_names


def identify_face(frame, known_encodings, known_names):
    """
    Detect and identify all faces in a frame.
    Returns a list of (top, right, bottom, left, name, distance) tuples.
    """
    small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small, model="hog", number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    results = []
    for encoding, location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        best_distance = distances[best_idx]

        name = known_names[best_idx] if best_distance < MATCH_THRESHOLD else "Unknown"

        inv_scale = int(1 / DETECTION_SCALE)
        top, right, bottom, left = [coord * inv_scale for coord in location]
        results.append((top, right, bottom, left, name, best_distance))

    return results


def draw_results(frame, stable_faces):
    """
    Draw bounding boxes and labels for all stable (confirmed) faces.
    stable_faces: dict of face_key -> {"location": ..., "name": ..., "distance": ...}
    """
    frame_height, frame_width = frame.shape[:2]

    for face_data in stable_faces.values():
        top, right, bottom, left = face_data["location"]
        name = face_data["name"]
        distance = face_data["distance"]

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        label = f"{name} ({distance:.2f})" if name != "Unknown" else "Unknown"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        label_y = bottom
        if bottom + text_height + 20 > frame_height:
            label_y = top - text_height - 10

        label_right = left + text_width + 10
        if label_right > frame_width:
            left = frame_width - text_width - 10
            label_right = frame_width

        cv2.rectangle(frame, (left, label_y), (label_right, label_y + text_height + 15), color, cv2.FILLED)
        cv2.putText(frame, label, (left + 5, label_y + text_height + 5),
                    font, font_scale, (255, 255, 255), thickness)


def _face_key(location):
    """Create a rough spatial key for a face location to track it across frames."""
    top, right, bottom, left = location
    # Bucket into grid cells so small movements don't create new keys
    cx = (left + right) // 2 // 120
    cy = (top + bottom) // 2 // 120
    return (cx, cy)


def run_recognition(on_identify=None, video_source=None):
    """
    Main recognition loop. Calls on_identify(name) for each newly confirmed face.
    Returns when the user presses 'q'.
    """
    known_encodings, known_names = load_encodings()

    if video_source is None:
        video_source = r"F:\Zain\Konversation\dev-mvp-face-detection\attendance-system\test-video\provided2.mp4"

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: could not open video source: {video_source}")
        sys.exit(1)

    frame_count = 0

    # pending[key] = {"name": str, "hits": int, "location": tuple, "distance": float}
    # Counts consecutive detections before confirming a face.
    pending = {}

    # stable[key] = {"name": str, "location": tuple, "distance": float, "hold": int}
    # Confirmed faces currently being drawn on screen.
    stable = {}

    # Track who has already been announced this session to avoid repeat callbacks.
    announced = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            frame_count += 1

            if frame_count % FRAME_SKIP == 0:
                raw_results = identify_face(frame, known_encodings, known_names)

                # Build a set of keys seen this detection pass
                seen_keys = set()

                for top, right, bottom, left, name, distance in raw_results:
                    location = (top, right, bottom, left)
                    key = _face_key(location)
                    seen_keys.add(key)

                    if key in pending and pending[key]["name"] == name:
                        # Same face, same name — increment hit counter
                        pending[key]["hits"] += 1
                        pending[key]["location"] = location
                        pending[key]["distance"] = distance
                    else:
                        # New face or name changed — reset
                        pending[key] = {"name": name, "hits": 1, "location": location, "distance": distance}

                    # Promote to stable once confirmed
                    if pending[key]["hits"] >= CONFIRM_HITS:
                        stable[key] = {
                            "name": name,
                            "location": location,
                            "distance": distance,
                            "hold": HOLD_FRAMES,
                        }
                        # Fire callback only once per name per session
                        if name != "Unknown" and name not in announced:
                            announced.add(name)
                            if on_identify:
                                on_identify(name)

                # Remove pending entries that weren't seen this pass
                for key in list(pending.keys()):
                    if key not in seen_keys:
                        del pending[key]

                # Refresh hold timer for stable faces still visible
                for key in seen_keys:
                    if key in stable:
                        stable[key]["hold"] = HOLD_FRAMES

            # Tick down hold timers and remove expired stable faces
            for key in list(stable.keys()):
                stable[key]["hold"] -= 1
                if stable[key]["hold"] <= 0:
                    del stable[key]

            draw_results(frame, stable)
            cv2.imshow("Attendance System - Press 'q' to exit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return list(announced)


if __name__ == "__main__":
    run_recognition()