# Face Detection Attendance System

A simple face detection and recognition-based attendance system built with Python, OpenCV, and the `face_recognition` library.

## Features
- Enroll new students with a single photo.
- Real-time face recognition from webcam or video file.
- Automatic attendance logging with duplicate prevention.
- Session summary upon exit.

## Setup

1. Install dependencies using [uv](https://github.com/astral/uv):
   ```bash
   uv sync
   ```

2. (Optional) Configure the video source in a `.env` file:
   ```env
   VIDEO_SOURCE=0  # Use 0 for webcam, or a path to a video file
   ```

## Usage

### 1. Enroll Students
Place images of students in the `known_faces/` directory and run:
```bash
python enroll.py --name "Student Name" --image "path/to/image.jpg"
```

### 2. Run Attendance System
```bash
python main.py
```
Press `q` to quit the application and see the session summary.

## Project Structure
- `enroll.py`: Script to add new faces to the system.
- `main.py`: Entry point for the attendance system.
- `recognize.py`: Core recognition logic.
- `logger.py`: Attendance logging logic.
- `data/`: Stores face encodings.
- `logs/`: Stores attendance records in JSON format.
- `known_faces/`: Suggested directory for storing student photos.
