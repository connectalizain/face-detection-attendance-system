# Attendance logging with duplicate prevention

import json
import os
from datetime import datetime


LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
ATTENDANCE_PATH = os.path.join(LOGS_DIR, "attendance.json")


def _load_logs():
    if not os.path.exists(ATTENDANCE_PATH):
        return []
    try:
        with open(ATTENDANCE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []


def _save_logs(logs):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(ATTENDANCE_PATH, "w") as f:
        json.dump(logs, f, indent=2)


def already_logged_today(name):
    today = datetime.now().strftime("%Y-%m-%d")
    logs = _load_logs()
    return any(entry["student"] == name and entry["date"] == today for entry in logs)


def log_attendance(name):
    """
    Log attendance for a student. Returns True if logged, False if already logged today.
    """
    if already_logged_today(name):
        return False

    now = datetime.now()
    entry = {
        "student": name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "status": "present",
    }

    logs = _load_logs()
    logs.append(entry)
    _save_logs(logs)

    print(f"Attendance logged: {name} at {entry['time']}")
    return True


def get_today_logs():
    """Return all attendance entries for today."""
    today = datetime.now().strftime("%Y-%m-%d")
    return [entry for entry in _load_logs() if entry["date"] == today]
