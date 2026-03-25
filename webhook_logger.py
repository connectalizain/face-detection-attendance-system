import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load a .env file with only these two variables: WEBHOOK_URL and CLASS_NAME
load_dotenv()

WEBHOOK_URL = os.getenv("WEBHOOK_URL")
CLASS_NAME = os.getenv("CLASS_NAME")

# Module-level announced set for duplicate prevention per session
announced = set()

def log_attendance_webhook(name, guardian_no, guardian_name, school_name):
    """
    Log attendance for a student via webhook. Returns True if logged, False if already logged this session.
    """
    if name in announced:
        return False

    announced.add(name)

    # Build the exact JSON payload
    payload = {
        "guardian_no": guardian_no,
        "guardian_name": guardian_name,
        "student_name": name,
        "school_name": school_name,
        "class_name": CLASS_NAME,
        "arrival_time": datetime.now().strftime("%H:%M:%S")
    }

    try:
        response = requests.post(WEBHOOK_URL, json=payload, headers={"Content-Type": "application/json"})
        if response.status_code == 200:
            print(f"Success: Attendance logged for {name}")
            return True
        else:
            print(f"Error logging attendance for {name}: Status Code {response.status_code}")
            return False
    except Exception as e:
        print(f"Exception logging attendance for {name}: {e}")
        return False
