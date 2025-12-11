# Smart Privacy-Aware AI/ML Patient Monitoring System

## Overview
A web-based patient monitoring system that uses AI/ML to detect falls, aggression, risky behavior, and emotions while protecting patient privacy. The system processes video frames in real-time without storing any images or faces.

## Project Structure
```
/
├── app.py              # Flask backend with API endpoints
├── ml_utils.py         # ML utilities for detection (pose, emotion, behavior)
├── static/
│   ├── css/
│   │   └── style.css   # Modern dashboard styling
│   └── js/
│       └── app.js      # Frontend webcam and UI logic
├── templates/
│   ├── index.html      # Live monitoring page
│   └── dashboard.html  # Dashboard overview
├── requirements.txt    # Python dependencies (auto-generated)
└── pyproject.toml      # Project configuration
```

## Tech Stack
- **Backend**: Python 3.11, Flask, Flask-CORS
- **AI/ML**: MediaPipe (pose estimation, face mesh), OpenCV, NumPy
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Architecture**: RESTful API with polling for alerts

## Key Features
1. **Fall Detection**: Uses MediaPipe pose estimation to detect body position anomalies
2. **Aggression Detection**: Analyzes arm velocity and gesture patterns
3. **Risky Behavior Detection**: Monitors body lean, stance stability, edge proximity
4. **Emotion Detection**: Uses facial landmarks to detect happy, sad, angry, scared, neutral
5. **Privacy Protection**: No frames stored, faces pixelated on display

## API Endpoints
- `GET /` - Live monitoring page
- `GET /dashboard` - Dashboard overview
- `POST /api/analyze` - Analyze frame (base64 image) and return detections
- `GET /api/alerts` - Get list of stored alerts
- `POST /api/alerts/clear` - Clear all alerts
- `GET /api/status` - Get system health status
- `POST /api/status/toggle` - Toggle monitoring on/off

## Running the Application
The application runs on port 5000 with Flask's development server:
```bash
python app.py
```

## Detection Logic
- **Fall**: Shoulders below hips, head near hip level, sudden vertical drops
- **Aggression**: High arm velocity + raised arms or forward motion
- **Risky**: Excessive body lean, unstable stance, near frame edges
- **Emotion**: Mouth shape, eye openness, eyebrow position analysis

## Recent Changes
- December 11, 2025: Initial project creation with full feature set
