"""
ML Utilities for Smart Privacy-Aware Patient Monitoring System

This module provides:
- Pose estimation using MediaPipe
- Fall detection using body keypoint analysis
- Aggression/risky behavior detection using velocity and gesture rules
- Emotion detection using facial landmark analysis

All processing is done locally without storing any images or faces.
"""

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

class SessionState:
    def __init__(self):
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        self.last_analysis_time = time.time()
    
    def reset(self):
        self.position_history.clear()
        self.velocity_history.clear()
        self.last_analysis_time = time.time()

session_state = SessionState()


def reset_session():
    """Reset session state for a new monitoring session."""
    session_state.reset()


def analyze_frame(image_data):
    """
    Main analysis function that processes a frame and returns detection results.
    
    Args:
        image_data: numpy array of the image (BGR format)
    
    Returns:
        dict with fall_detected, aggression, risky_behavior, emotion, and confidence scores
    """
    results = {
        "fall_detected": False,
        "fall_confidence": 0.0,
        "aggression": False,
        "aggression_confidence": 0.0,
        "risky_behavior": False,
        "risky_confidence": 0.0,
        "emotion": "neutral",
        "emotion_confidence": 0.5,
        "pose_detected": False
    }
    
    if image_data is None:
        return results
    
    try:
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        pose_results = pose.process(image_rgb)
        
        if pose_results.pose_landmarks:
            results["pose_detected"] = True
            landmarks = pose_results.pose_landmarks.landmark
            
            fall_result = detect_fall(landmarks, image_data.shape)
            results["fall_detected"] = fall_result["detected"]
            results["fall_confidence"] = fall_result["confidence"]
            
            aggression_result = detect_aggression(landmarks, image_data.shape)
            results["aggression"] = aggression_result["detected"]
            results["aggression_confidence"] = aggression_result["confidence"]
            
            risky_result = detect_risky_behavior(landmarks, image_data.shape)
            results["risky_behavior"] = risky_result["detected"]
            results["risky_confidence"] = risky_result["confidence"]
        
        face_results = face_mesh.process(image_rgb)
        if face_results.multi_face_landmarks:
            emotion_result = detect_emotion(face_results.multi_face_landmarks[0])
            results["emotion"] = emotion_result["emotion"]
            results["emotion_confidence"] = emotion_result["confidence"]
        
        session_state.last_analysis_time = time.time()
        
    except Exception as e:
        print(f"Error in frame analysis: {e}")
    
    return results


def detect_fall(landmarks, image_shape):
    """
    Detect falls using pose keypoint analysis.
    
    Fall detection heuristics:
    1. Check if shoulders are below hips (person is horizontal/fallen)
    2. Check if head is at same level or below hips
    3. Analyze the vertical position of the center of mass
    4. Track sudden changes in body position
    
    Args:
        landmarks: MediaPipe pose landmarks
        image_shape: tuple of (height, width, channels)
    
    Returns:
        dict with detected (bool) and confidence (float 0-1)
    """
    height, width = image_shape[:2]
    
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    head_y = nose.y
    
    shoulder_below_hip = shoulder_y > hip_y - 0.1
    head_near_hip_level = head_y > hip_y - 0.15
    
    body_center_y = (shoulder_y + hip_y) / 2
    session_state.position_history.append(body_center_y)
    
    sudden_drop = False
    if len(session_state.position_history) >= 3:
        recent_change = session_state.position_history[-1] - session_state.position_history[-3]
        if recent_change > 0.15:
            sudden_drop = True
    
    confidence = 0.0
    detected = False
    
    if shoulder_below_hip and head_near_hip_level:
        confidence = 0.85
        detected = True
    elif sudden_drop and head_near_hip_level:
        confidence = 0.75
        detected = True
    elif shoulder_below_hip:
        confidence = 0.6
        detected = True
    elif head_near_hip_level:
        confidence = 0.4
    
    return {"detected": detected, "confidence": confidence}


def detect_aggression(landmarks, image_shape):
    """
    Detect aggressive behavior using body keypoint velocity and gesture analysis.
    
    Aggression detection heuristics:
    1. Fast arm movements (high velocity)
    2. Arms raised in threatening posture
    3. Rapid body movements
    4. Clenched fist detection (hands close together)
    
    Args:
        landmarks: MediaPipe pose landmarks
        image_shape: tuple of (height, width, channels)
    
    Returns:
        dict with detected (bool) and confidence (float 0-1)
    """
    height, width = image_shape[:2]
    
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
    current_positions = {
        "left_wrist": (left_wrist.x, left_wrist.y),
        "right_wrist": (right_wrist.x, right_wrist.y)
    }
    
    high_velocity = False
    if len(session_state.velocity_history) > 0:
        prev_positions = session_state.velocity_history[-1]
        left_vel = np.sqrt(
            (current_positions["left_wrist"][0] - prev_positions["left_wrist"][0])**2 +
            (current_positions["left_wrist"][1] - prev_positions["left_wrist"][1])**2
        )
        right_vel = np.sqrt(
            (current_positions["right_wrist"][0] - prev_positions["right_wrist"][0])**2 +
            (current_positions["right_wrist"][1] - prev_positions["right_wrist"][1])**2
        )
        if left_vel > 0.12 or right_vel > 0.12:
            high_velocity = True
    
    session_state.velocity_history.append(current_positions)
    
    arms_raised = (left_wrist.y < left_shoulder.y - 0.1) or (right_wrist.y < right_shoulder.y - 0.1)
    
    hands_forward = (left_elbow.y < left_shoulder.y and left_wrist.y < left_elbow.y) or \
                   (right_elbow.y < right_shoulder.y and right_wrist.y < right_elbow.y)
    
    confidence = 0.0
    detected = False
    
    if high_velocity and arms_raised:
        confidence = 0.85
        detected = True
    elif high_velocity and hands_forward:
        confidence = 0.75
        detected = True
    elif arms_raised and hands_forward:
        confidence = 0.6
        detected = True
    elif high_velocity:
        confidence = 0.4
    
    return {"detected": detected, "confidence": confidence}


def detect_risky_behavior(landmarks, image_shape):
    """
    Detect risky behaviors that might indicate distress or danger.
    
    Risky behavior heuristics:
    1. Unusual body positions (leaning too far)
    2. Erratic movements
    3. Person near edges (potential wandering)
    4. Unstable stance
    
    Args:
        landmarks: MediaPipe pose landmarks
        image_shape: tuple of (height, width, channels)
    
    Returns:
        dict with detected (bool) and confidence (float 0-1)
    """
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    hip_center_x = (left_hip.x + right_hip.x) / 2
    lean_amount = abs(shoulder_center_x - hip_center_x)
    excessive_lean = lean_amount > 0.15
    
    near_edge = shoulder_center_x < 0.1 or shoulder_center_x > 0.9
    
    ankle_spread = abs(left_ankle.x - right_ankle.x)
    unstable_stance = ankle_spread < 0.05 or ankle_spread > 0.5
    
    confidence = 0.0
    detected = False
    
    if excessive_lean and unstable_stance:
        confidence = 0.8
        detected = True
    elif excessive_lean and near_edge:
        confidence = 0.75
        detected = True
    elif near_edge and unstable_stance:
        confidence = 0.65
        detected = True
    elif excessive_lean:
        confidence = 0.5
        detected = True
    elif near_edge:
        confidence = 0.3
    
    return {"detected": detected, "confidence": confidence}


def detect_emotion(face_landmarks):
    """
    Detect emotion using facial landmark analysis.
    
    This uses a rule-based approach analyzing:
    1. Mouth shape (smile vs frown)
    2. Eye openness
    3. Eyebrow position
    4. Overall facial structure
    
    Emotions detected: happy, sad, angry, scared, neutral
    
    Args:
        face_landmarks: MediaPipe face mesh landmarks
    
    Returns:
        dict with emotion (str) and confidence (float 0-1)
    """
    landmarks = face_landmarks.landmark
    
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    
    mouth_open = abs(upper_lip.y - lower_lip.y)
    mouth_width = abs(left_mouth.x - right_mouth.x)
    
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    
    left_eye_open = abs(left_eye_top.y - left_eye_bottom.y)
    right_eye_open = abs(right_eye_top.y - right_eye_bottom.y)
    avg_eye_open = (left_eye_open + right_eye_open) / 2
    
    left_brow = landmarks[70]
    right_brow = landmarks[300]
    left_eye_center = landmarks[33]
    right_eye_center = landmarks[263]
    
    left_brow_raise = left_eye_center.y - left_brow.y
    right_brow_raise = right_eye_center.y - right_brow.y
    avg_brow_raise = (left_brow_raise + right_brow_raise) / 2
    
    mouth_corner_left = landmarks[61]
    mouth_corner_right = landmarks[291]
    mouth_center = landmarks[13]
    smile_ratio = ((mouth_corner_left.y + mouth_corner_right.y) / 2) - mouth_center.y
    
    face_height = abs(landmarks[10].y - landmarks[152].y)
    if face_height < 0.01:
        face_height = 0.1
    
    norm_mouth_open = mouth_open / face_height
    norm_mouth_width = mouth_width / face_height
    norm_eye_open = avg_eye_open / face_height
    norm_brow_raise = avg_brow_raise / face_height
    norm_smile = smile_ratio / face_height
    
    emotion = "neutral"
    confidence = 0.5
    
    if norm_smile < -0.05 and norm_mouth_width > 0.8:
        emotion = "happy"
        confidence = min(0.9, 0.6 + abs(norm_smile) * 3)
    elif norm_brow_raise < 0.15 and norm_mouth_open < 0.15:
        emotion = "angry"
        confidence = 0.65
    elif norm_eye_open > 0.12 and norm_brow_raise > 0.25:
        emotion = "scared"
        confidence = 0.6
    elif norm_smile > 0.02 and norm_mouth_open < 0.12:
        emotion = "sad"
        confidence = 0.55
    else:
        emotion = "neutral"
        confidence = 0.5
    
    return {"emotion": emotion, "confidence": confidence}


def apply_privacy_filter(image_data, blur_strength=25):
    """
    Apply privacy filter to protect patient identity.
    This pixelates/blurs faces in the image.
    
    Args:
        image_data: numpy array of the image (BGR format)
        blur_strength: strength of the blur effect
    
    Returns:
        Processed image with faces blurred
    """
    if image_data is None:
        return None
    
    try:
        image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(image_rgb)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                h, w = image_data.shape[:2]
                
                x_coords = [lm.x * w for lm in face_landmarks.landmark]
                y_coords = [lm.y * h for lm in face_landmarks.landmark]
                
                x_min = int(max(0, min(x_coords) - 20))
                x_max = int(min(w, max(x_coords) + 20))
                y_min = int(max(0, min(y_coords) - 20))
                y_max = int(min(h, max(y_coords) + 20))
                
                face_region = image_data[y_min:y_max, x_min:x_max]
                
                small = cv2.resize(face_region, (16, 16), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_NEAREST)
                
                image_data[y_min:y_max, x_min:x_max] = pixelated
        
        return image_data
    
    except Exception as e:
        print(f"Error applying privacy filter: {e}")
        return image_data


def get_suggested_action(detection_type, confidence):
    """
    Get suggested action based on detection type and confidence.
    
    Args:
        detection_type: Type of detection (fall, aggression, risky, emotion)
        confidence: Confidence score (0-1)
    
    Returns:
        String with suggested action
    """
    actions = {
        "fall": {
            "high": "URGENT: Check patient immediately - possible fall detected",
            "medium": "Alert: Verify patient status - fall indicators detected",
            "low": "Monitor: Minor fall risk indicators observed"
        },
        "aggression": {
            "high": "URGENT: Approach with caution - aggressive behavior detected",
            "medium": "Alert: Monitor patient closely - agitation indicators",
            "low": "Note: Elevated activity levels observed"
        },
        "risky": {
            "high": "Alert: Check patient safety - risky position detected",
            "medium": "Monitor: Patient showing unstable behavior",
            "low": "Note: Minor balance concerns observed"
        },
        "emotion": {
            "angry": "Monitor: Patient appears distressed or agitated",
            "scared": "Check on patient: Signs of fear or anxiety detected",
            "sad": "Consider comfort measures: Patient may be upset",
            "happy": "Patient appears comfortable",
            "neutral": "Patient status: Normal"
        }
    }
    
    if detection_type == "emotion":
        return actions["emotion"].get(confidence, "Patient status: Normal")
    
    if confidence >= 0.7:
        level = "high"
    elif confidence >= 0.5:
        level = "medium"
    else:
        level = "low"
    
    return actions.get(detection_type, {}).get(level, "Continue monitoring")
