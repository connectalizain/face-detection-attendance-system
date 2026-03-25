# Enrolls a new student from a photo into encodings.pkl

import argparse
import os
import pickle
import sys

import face_recognition


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ENCODINGS_PATH = os.path.join(DATA_DIR, "encodings.pkl")


def load_encodings():
    if not os.path.exists(ENCODINGS_PATH):
        return []
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error: could not read encodings file: {e}")
        sys.exit(1)


def save_encodings(encodings):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)


def enroll_student(name, guardian_no="", guardian_name="", school_name="", image_path=""):
    if not image_path or not os.path.exists(image_path):
        print(f"Error: image file not found: {image_path}")
        sys.exit(1)

    try:
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        print(f"Error: could not read image: {e}")
        sys.exit(1)

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        print("Error: no face detected in the image. Check lighting and ensure the face is clearly visible.")
        sys.exit(1)

    if len(face_locations) > 1:
        print(f"Warning: {len(face_locations)} faces found in image. Using only the first face.")

    encodings = face_recognition.face_encodings(image, known_face_locations=[face_locations[0]])
    encoding = encodings[0]

    all_encodings = load_encodings()
    all_encodings.append({
        "name": name,
        "guardian_no": guardian_no,
        "guardian_name": guardian_name,
        "school_name": school_name,
        "encoding": encoding
    })
    save_encodings(all_encodings)

    unique_students = len({e["name"] for e in all_encodings})
    print(f"Enrolled {name} successfully. Total students: {unique_students}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a student into the face recognition system.")
    parser.add_argument("--name", required=True, help="Student name")
    parser.add_argument("--image", required=True, help="Path to the student's photo")
    parser.add_argument("--guardian_no", default="", help="Guardian phone number (optional)")
    parser.add_argument("--guardian_name", default="", help="Guardian name (optional)")
    parser.add_argument("--school_name", default="", help="School name (optional)")
    args = parser.parse_args()

    enroll_student(args.name, args.guardian_no, args.guardian_name, args.school_name, args.image)
