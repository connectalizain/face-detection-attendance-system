import os
import pickle
import sys
import cv2
import face_recognition
import numpy as np

ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "encodings.pkl")
MATCH_THRESHOLD = 0.55  # Slightly lenient for angled faces
FRAME_SKIP = 2          # Run recognition every 2nd frame
DETECTION_SCALE = 0.4   # Good balance of speed vs detail for HOG

# --- Stability Settings ---
CONFIRMATION_THRESHOLD = 2  # Hits needed to confirm
PERSISTENCE_LIMIT = 20      # Frames to remember a face (increased for smoother display)

def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        print(f"Error: encodings file not found.")
        sys.exit(1)
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return (
        [e["encoding"] for e in data],
        [e["name"] for e in data],
        [e.get("guardian_no", "N/A") for e in data],
        [e.get("guardian_name", "N/A") for e in data],
        [e.get("school_name", "N/A") for e in data]
    )

def preprocess_frame(frame):
    """
    Improve frame quality before detection.
    Equalizes contrast per channel to help with motion blur and poor lighting.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def identify_face(frame, known_encodings, known_names, known_guardian_nos, known_guardian_names, known_school_names):
    # 1. Preprocess for better detection
    enhanced = preprocess_frame(frame)

    # 2. Resize for speed
    small_frame = cv2.resize(enhanced, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 3. Detect faces — upsample=2 catches smaller and angled faces
    face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=2)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=1)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back up to original frame size
        top, right, bottom, left = [int(v / DETECTION_SCALE) for v in [top, right, bottom, left]]

        distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        best_dist = 1.0
        g_no, g_name, s_name = "N/A", "N/A", "N/A"

        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            if distances[best_match_index] <= MATCH_THRESHOLD:
                name = known_names[best_match_index]
                best_dist = distances[best_match_index]
                g_no = known_guardian_nos[best_match_index]
                g_name = known_guardian_names[best_match_index]
                s_name = known_school_names[best_match_index]

        results.append((top, right, bottom, left, name, best_dist, g_no, g_name, s_name))

    # 4. Overlap filter — keeps best match when two boxes overlap the same face
    final_results = []
    results.sort(key=lambda x: x[5])  # Sort by distance (best first)
    for res in results:
        is_duplicate = False
        for final in final_results:
            center_dist = np.sqrt(((res[0] - final[0]) ** 2) + ((res[3] - final[3]) ** 2))
            if center_dist < 60:  # Slightly larger threshold for angled faces
                is_duplicate = True
                break
        if not is_duplicate:
            final_results.append(res)

    return final_results

def draw_results(frame, faces, tracker):
    for (top, right, bottom, left, name, dist, *_) in faces:
        color = (0, 0, 255)  # Red = unconfirmed
        if name in tracker and tracker[name]["count"] >= CONFIRMATION_THRESHOLD:
            color = (0, 255, 0)  # Green = confirmed

        # Bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label
        label = f"{name} ({dist:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position label below face box by default
        label_x = left
        label_rect_y1 = bottom
        label_rect_y2 = bottom + label_height + baseline + 10
        label_text_y = bottom + label_height + 5

        # If label goes off bottom, put it above
        if label_rect_y2 > frame.shape[0]:
            label_rect_y1 = top - label_height - baseline - 10
            label_rect_y2 = top
            label_text_y = top - baseline - 5

        # If label goes off right edge, shift left
        if label_x + label_width > frame.shape[1]:
            label_x = frame.shape[1] - label_width - 5

        # Filled label rectangle + white text
        cv2.rectangle(frame, (label_x, label_rect_y1), (label_x + label_width, label_rect_y2), color, cv2.FILLED)
        cv2.putText(frame, label, (label_x, label_text_y), font, font_scale, (255, 255, 255), thickness)

def run_recognition(video_source=None, on_identify=None):
    known_encodings, known_names, known_guardian_nos, known_guardian_names, known_school_names = load_encodings()
    cap = cv2.VideoCapture(video_source or 0)

    tracker = {}
    announced = set()
    current_faces = []

    # Respect actual video FPS so it doesn't play too fast
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30
    wait_ms = max(1, min(100, int(1000 / fps)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP == 0:
            current_faces = identify_face(
                frame, known_encodings, known_names,
                known_guardian_nos, known_guardian_names, known_school_names
            )

            for face in current_faces:
                name = face[4]
                if name == "Unknown":
                    continue

                if name not in tracker:
                    tracker[name] = {
                        "count": 0,
                        "persistence": PERSISTENCE_LIMIT,
                        "guardian_no": face[6],
                        "guardian_name": face[7],
                        "school_name": face[8]
                    }
                tracker[name]["count"] += 1
                tracker[name]["persistence"] = PERSISTENCE_LIMIT

                if tracker[name]["count"] >= CONFIRMATION_THRESHOLD and name not in announced:
                    if on_identify:
                        on_identify(
                            name,
                            tracker[name]["guardian_no"],
                            tracker[name]["guardian_name"],
                            tracker[name]["school_name"]
                        )
                    announced.add(name)

        # Always draw last known faces (persists across skipped frames)
        draw_results(frame, current_faces, tracker)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()