# Bulk enrolls all students from the known_faces directory

import os
import pickle
import sys
import face_recognition

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ENCODINGS_PATH = os.path.join(DATA_DIR, "encodings.pkl")
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")

def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        return []
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error: could not read encodings file: {e}")
        return []

def save_encodings(encodings):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)
    print(f"Saved {len(encodings)} encodings to {ENCODINGS_PATH}")

def enroll_all():
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: {KNOWN_FACES_DIR} not found.")
        sys.exit(1)

    all_encodings = []
    
    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"Enrolling {person_name}...")
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                image = face_recognition.load_image_file(image_path)
                # Use cnn model for enrollment if possible for better quality encodings
                # But hog is fine for enrollment if images are clear
                face_locations = face_recognition.face_locations(image)
                
                if len(face_locations) == 0:
                    print(f"  [!] No face found in {image_name}. Skipping.")
                    continue
                
                encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
                if encodings:
                    all_encodings.append({"name": person_name, "encoding": encodings[0]})
                    print(f"  [+] Enrolled {image_name}")
            except Exception as e:
                print(f"  [!] Error processing {image_name}: {e}")

    save_encodings(all_encodings)
    unique_students = len({e["name"] for e in all_encodings})
    print(f"Bulk enrollment complete. Total images: {len(all_encodings)}, Total students: {unique_students}")

if __name__ == "__main__":
    enroll_all()
