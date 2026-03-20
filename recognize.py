# Core face recognition logic

import os
import pickle
import sys

import cv2
import face_recognition
import numpy as np


ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "encodings.pkl")
MATCH_THRESHOLD = 0.6
FRAME_SKIP = 5


def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        print(f"Error: encodings file not found at {ENCODINGS_PATH}. Run enroll.py first.")
        sys.exit(1)

    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)

    if not data:
        print("Error: encodings file is empty. Enroll at least one student first.")
        sys.exit(1)

    known_encodings = [entry["encoding"] for entry in data]
    known_names = [entry["name"] for entry in data]
    return known_encodings, known_names


def identify_face(frame, known_encodings, known_names):
    """
    Detect and identify all faces in a frame.
    Returns a list of (top, right, bottom, left, name) tuples in full-frame coordinates.
    """
    small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    results = []
    for encoding, location in zip(face_encodings, face_locations):
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_idx = int(np.argmin(distances))

        if distances[best_idx] < MATCH_THRESHOLD:
            name = known_names[best_idx]
        else:
            name = "Unknown"

        # Scale coordinates back to full frame size
        top, right, bottom, left = [coord * 5 for coord in location]
        results.append((top, right, bottom, left, name))

    return results


def draw_results(frame, results):
    for top, right, bottom, left, name in results:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom), (right, bottom + 24), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 4, bottom + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def run_recognition(on_identify=None):
    """
    Main recognition loop. Calls on_identify(name) for each identified face.
    Returns when the user presses 'q'.
    """
    known_encodings, known_names = load_encodings()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(r"F:\Zain\Konversation\dev-mvp-face-detection\attendance-system\test-video\ali.mp4")
    if not cap.isOpened():
        print("Error: could not open webcam.")
        sys.exit(1)

    frame_count = 0
    last_results = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from webcam.")
                break

            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                last_results = identify_face(frame, known_encodings, known_names)
                if on_identify:
                    for _, _, _, _, name in last_results:
                        on_identify(name)

            draw_results(frame, last_results)
            cv2.imshow("Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return [name for _, _, _, _, name in last_results]


if __name__ == "__main__":
    run_recognition()
