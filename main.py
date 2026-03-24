# Entry point: runs the main camera loop

import sys
from datetime import datetime

from logger import get_today_logs, log_attendance
from recognize import run_recognition


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
        # Deduplicate names for the summary
        present_students = {entry['student'] for entry in logs}
        for student in sorted(present_students):
            # Find the first log for this student today
            first_log = next(entry for entry in logs if entry['student'] == student)
            print(f"  ✓ {student} — {first_log['time']}")
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
