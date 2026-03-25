# Entry point: runs the main camera loop

import sys
from datetime import datetime

from webhook_logger import log_attendance_webhook, announced
from recognize import run_recognition


def on_identify(name, guardian_no, guardian_name, school_name):
    if name == "Unknown":
        return
    log_attendance_webhook(name, guardian_no, guardian_name, school_name)


def print_summary():
    print("\n--- Session Summary ---")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    if announced:
        for student in sorted(announced):
            print(f"  ✓ {student}")
    else:
        print("  No attendance logged this session")


if __name__ == "__main__":
    # Allow passing video path as first argument
    # e.g. python main.py 0 (for webcam)
    video_source = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Check if we should use integer 0 for webcam
    if video_source == "0":
        video_source = 0

    print("Attendance System Starting...")
    print("Press 'q' to quit")

    try:
        run_recognition(on_identify=on_identify, video_source=video_source)
    except KeyboardInterrupt:
        pass

    print_summary()
