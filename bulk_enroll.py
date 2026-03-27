import os
print("Running from:", os.getcwd())
from enroll import enroll_student, ENCODINGS_PATH

# Delete data/encodings.pkl if it exists before starting fresh
if os.path.exists(ENCODINGS_PATH):
    os.remove(ENCODINGS_PATH)
    print(f"Deleted existing encodings at {ENCODINGS_PATH}")

STUDENTS = [
     {"name": "secretario lucas", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image.jpeg"]},
    {"name": "secretario aldo", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image.jpeg"]},
    {"name": "Bento Oliveira", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image.jpeg"]},
    {"name": "Cavalcanti", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image.jpeg"]},
    {"name": "Evania Galdino", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image.jpeg"]},
    {"name": "Frederico", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image1.jpeg", "image.jpeg"]},
    {"name": "Freire", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image.jpeg"]},
    {"name": "girl", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image.jpeg"]},
    {"name": "Valtinho Araújo", "guardian_no": "+558481335076", "guardian_name": "Frederico Lima", "school_name": "ABC School", "images": ["image.jpeg", "image1.jpeg" ]},
   
]

total_students = 0
total_images = 0

for student in STUDENTS:
    name = student["name"]
    g_no = student["guardian_no"]
    g_name = student["guardian_name"]
    s_name = student["school_name"]
    
    enrolled_at_least_one = False
    for image_filename in student["images"]:
        image_path = os.path.join("known_faces", name, image_filename)
        
        if not os.path.exists(image_path):
            print(f"Warning: image file not found, skipping: {image_path}")
            continue
            
        enroll_student(name, g_no, g_name, s_name, image_path)
        total_images += 1
        enrolled_at_least_one = True
    
    if enrolled_at_least_one:
        total_students += 1

print("\n--- Bulk Enrollment Summary ---")
print(f"Total students successfully enrolled: {total_students}")
print(f"Total images processed: {total_images}")
