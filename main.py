# Entry point: runs the main camera loop

from datetime import datetime

from logger import get_today_logs, log_attendance
from recognize import load_encodings, run_recognition


def on_identify(name):
    if name == "Unknown":
        return
    if log_attendance(name):
        print(f"✓ {name} marked present")


def print_summary():
    logs = get_today_logs()
    print("\n--- Session Summary ---")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    if logs:
        for entry in logs:
            print(f"  ✓ {entry['student']} — {entry['time']}")
    else:
        print("  No attendance logged this session")


if __name__ == "__main__":
    known_encodings, _ = load_encodings()
    print("Attendance System Starting...")
    print(f"Loaded {len(known_encodings)} known faces")
    print("Press 'q' to quit")

    try:
        run_recognition(on_identify=on_identify)
    except KeyboardInterrupt:
        pass

    print_summary()
