import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np


app = Flask(__name__)

# Initialize MediaPipe Pose Detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables for rep counting
counter = 0
stage = None

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Function to reset counter and stage
def reset_counters():
    global counter, stage
    counter = 0
    stage = None

## Arm Curl Posture Detection
def check_arm_curl_posture(landmarks):
    global counter, stage
    feedback = ""

    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    # Only increment counter when stage transitions from 'down' to 'up'
    if elbow_angle > 160:
        stage = "down"
        feedback = "Keep your arms straight!"
    elif elbow_angle < 30 and stage == "down":
        stage = "up"
        counter += 1  # Count only when the full transition completes
        feedback = f"Good job! Reps: {counter}"

    return feedback


# Pushup Posture Detection
def check_pushup_posture(landmarks):
    global counter, stage
    feedback = ""

    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    body_angle = calculate_angle(shoulder, hip, ankle)

    if elbow_angle > 160:
        stage = "up"
        feedback = "Lower your body!"
    elif elbow_angle < 90 and stage == "up":
        stage = "down"
        counter += 1  # Count only if posture is correct
        feedback = "Great! Push up again!"

    if body_angle < 160:
        feedback += " Keep your body straight!"

    return feedback

# Weightlifting Posture Detection
def check_weightlifting_posture(landmarks):
    global counter, stage
    feedback = ""

    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    if elbow_angle > 160:
        stage = "up"
        feedback = "Lower the weights!"
    elif elbow_angle < 90 and stage == "up":
        stage = "down"
        counter += 1  # Count only if posture is correct
        feedback = "Excellent! Lift again!"

    return feedback

def gen_frames():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Call posture checking functions here
                arm_curl_feedback = check_arm_curl_posture(results.pose_landmarks.landmark)
                pushup_feedback = check_pushup_posture(results.pose_landmarks.landmark)
                weightlifting_feedback = check_weightlifting_posture(results.pose_landmarks.landmark)

                # Display the counter
                cv2.putText(frame, f'Reps: {counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Display feedback messages
                if arm_curl_feedback:
                    cv2.putText(frame, arm_curl_feedback, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                if pushup_feedback:
                    cv2.putText(frame, pushup_feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                if weightlifting_feedback:
                    cv2.putText(frame, weightlifting_feedback, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    reset_counters()  # Reset counters on index page
    return render_template('index.html')

@app.route('/exercise/<exercise_name>')
def exercise(exercise_name):
    return render_template('exercise.html', exercise_name=exercise_name, video_feed_url="/video_feed")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
