import os
import pickle
import sys
import cv2
import face_recognition
import numpy as np

ENCODINGS_PATH = os.path.join(os.path.dirname(__file__), "data", "encodings.pkl")
MATCH_THRESHOLD = 0.5  # Stricter as per your requirements
FRAME_SKIP = 3         # Run recognition every 3rd frame
DETECTION_SCALE = 0.25 # Faster processing

# --- Stability Settings ---
CONFIRMATION_THRESHOLD = 3  # Hits needed to show "Green"
PERSISTENCE_LIMIT = 15      # Frames to remember a face after it's gone

def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        print(f"Error: encodings file not found.")
        sys.exit(1)
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return [e["encoding"] for e in data], [e["name"] for e in data]

def identify_face(frame, known_encodings, known_names):
    # 1. Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 2. Detect and Encode
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back up
        top, right, bottom, left = [int(v / DETECTION_SCALE) for v in [top, right, bottom, left]]
        
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        best_dist = 1.0

        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            if distances[best_match_index] <= MATCH_THRESHOLD:
                name = known_names[best_match_index]
                best_dist = distances[best_match_index]

        results.append((top, right, bottom, left, name, best_dist))
    
    # 3. Simple Overlap Filter (Prevents Double Boxes)
    # If two boxes are mostly overlapping, keep the one with the better distance
    final_results = []
    results.sort(key=lambda x: x[5]) # Sort by best distance
    for res in results:
        is_duplicate = False
        for final in final_results:
            # Check if centers are close (simple heuristic)
            center_dist = np.sqrt(((res[0]-final[0])**2) + ((res[3]-final[3])**2))
            if center_dist < 50: # If boxes are within 50px of each other
                is_duplicate = True
                break
        if not is_duplicate:
            final_results.append(res)
            
    return final_results

def run_recognition(video_source=None, on_identify=None):
    known_encodings, known_names = load_encodings()
    cap = cv2.VideoCapture(video_source or 0)
    
    # tracker: { "Name": {"count": 0, "persistence": 0} }
    tracker = {}
    announced = set()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Run detection every N frames
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % FRAME_SKIP == 0:
            current_faces = identify_face(frame, known_encodings, known_names)
            
            # Update tracker
            active_names = [f[4] for f in current_faces if f[4] != "Unknown"]
            for name in active_names:
                if name not in tracker:
                    tracker[name] = {"count": 0, "persistence": PERSISTENCE_LIMIT}
                tracker[name]["count"] += 1
                tracker[name]["persistence"] = PERSISTENCE_LIMIT # Reset persistence
                
                # Check if confirmed
                if tracker[name]["count"] >= CONFIRMATION_THRESHOLD and name not in announced:
                    if on_identify: on_identify(name)
                    announced.add(name)

        # Draw Logic
        for (top, right, bottom, left, name, dist) in (current_faces if 'current_faces' in locals() else []):
            color = (0, 0, 255) # Red for Unknown/Unconfirmed
            if name in tracker and tracker[name]["count"] >= CONFIRMATION_THRESHOLD:
                color = (0, 255, 0) # Green for confirmed
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"{name} ({dist:.2f})", (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()