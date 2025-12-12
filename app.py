"""
Smart Privacy-Aware AI/ML Patient Monitoring System - Flask Backend

This application provides:
- Real-time patient monitoring via webcam
- Fall detection, aggression detection, and emotion recognition
- Privacy-first approach: no frames or faces stored
- RESTful API for frontend communication
"""

import os
import base64
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ml_utils import analyze_frame, apply_privacy_filter, get_suggested_action, reset_session

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
CORS(app)

alerts_store = []
MAX_ALERTS = 100

system_status = {
    "status": "healthy",
    "last_check": datetime.now().isoformat(),
    "monitoring_active": False,
    "total_events_today": 0,
    "last_emotion": "neutral"
}


@app.route('/')
def index():
    """Render the main monitoring page."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Render the dashboard overview page."""
    return render_template('dashboard.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze a video frame for patient monitoring.
    
    Expects JSON with:
    - image: base64 encoded image data
    
    Returns:
    - fall_detected: bool
    - aggression: bool
    - risky_behavior: bool
    - emotion: str
    - confidence scores for each detection
    """
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data provided"}), 400
        
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400
        
        results = analyze_frame(frame)
        
        system_status["monitoring_active"] = True
        system_status["last_check"] = datetime.now().isoformat()
        system_status["last_emotion"] = results.get("emotion", "neutral")
        
        if results.get("fall_detected"):
            add_alert("Fall", results.get("fall_confidence", 0), 
                     get_suggested_action("fall", results.get("fall_confidence", 0)))
        
        if results.get("aggression"):
            add_alert("Aggression", results.get("aggression_confidence", 0),
                     get_suggested_action("aggression", results.get("aggression_confidence", 0)))
        
        if results.get("risky_behavior"):
            add_alert("Risky Behavior", results.get("risky_confidence", 0),
                     get_suggested_action("risky", results.get("risky_confidence", 0)))
        
        emotion = results.get("emotion", "neutral")
        if emotion in ["angry", "scared", "sad"]:
            add_alert(f"Emotion: {emotion.capitalize()}", results.get("emotion_confidence", 0),
                     get_suggested_action("emotion", emotion))
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in analyze endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """
    Get list of stored alerts.
    
    Query params:
    - limit: max number of alerts to return (default 50)
    
    Returns list of alert objects with timestamp, issue, confidence, action.
    """
    limit = request.args.get('limit', 50, type=int)
    return jsonify(alerts_store[-limit:])


@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all stored alerts."""
    global alerts_store
    alerts_store = []
    system_status["total_events_today"] = 0
    return jsonify({"success": True, "message": "Alerts cleared"})


@app.route('/api/status', methods=['GET'])
def get_status():
    """
    Get system health status.
    
    Returns:
    - status: system health (healthy/warning/error)
    - last_check: last analysis timestamp
    - monitoring_active: whether monitoring is running
    - total_events_today: count of events today
    - last_emotion: most recently detected emotion
    """
    return jsonify(system_status)


@app.route('/api/status/toggle', methods=['POST'])
def toggle_monitoring():
    """Toggle monitoring status on/off."""
    system_status["monitoring_active"] = not system_status["monitoring_active"]
    
    if system_status["monitoring_active"]:
        reset_session()
    
    return jsonify({
        "monitoring_active": system_status["monitoring_active"],
        "message": "Monitoring " + ("started" if system_status["monitoring_active"] else "stopped")
    })


def add_alert(issue_type, confidence, action):
    """
    Add a new alert to the store.
    
    Args:
        issue_type: Type of issue detected
        confidence: Confidence score (0-1)
        action: Suggested action
    """
    global alerts_store
    
    alert = {
        "id": len(alerts_store) + 1,
        "timestamp": datetime.now().isoformat(),
        "issue": issue_type,
        "confidence": round(confidence, 2),
        "action": action
    }
    
    alerts_store.append(alert)
    system_status["total_events_today"] += 1
    
    if len(alerts_store) > MAX_ALERTS:
        alerts_store = alerts_store[-MAX_ALERTS:]


@app.after_request
def add_header(response):
    """Add headers to disable caching for development."""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
@app.route('/upload_video', methods=['POST'])
def upload_video():
    try:
        video_file = request.files.get("video")
        if not video_file:
            return jsonify({"error": "No video file provided"}), 400

        # Save uploaded video
        video_path = "uploaded_video.mp4"
        video_file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_rate = int(fps)  # analyze 1 frame per second

        results_summary = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # analyze only every 1 second
            if frame_index % sample_rate == 0:
                result = analyze_frame(frame)
                results_summary.append(result)

            frame_index += 1

        cap.release()

        return jsonify({
            "ok": True,
            "samples_analyzed": len(results_summary),
            "results": results_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500





if __name__ == '__main__':
    print("\n🔗 Your project is running at: http://127.0.0.1:5000/\n")
    app.run(host='0.0.0.0', port=5000, debug=False)


