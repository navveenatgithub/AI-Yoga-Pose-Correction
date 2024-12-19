import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize OpenCV for webcam feed
cap = cv2.VideoCapture(0)

# Utility function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Second point (vertex)
    c = np.array(c)  # Third point
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle

    return angle

# Function to extract angles (as you have already created)
def check_visibility(landmarks, indices):
    return all(landmarks[i].visibility > 0.5 for i in indices)

def calculate_joint_angle(landmarks, point1, point2, point3):
    return calculate_angle(
        [landmarks[point1].x, landmarks[point1].y],
        [landmarks[point2].x, landmarks[point2].y],
        [landmarks[point3].x, landmarks[point3].y]
    )

def extract_angles(landmarks):
    # Define the angles to calculate and the respective landmark indices
    angle_definitions = {
        # Elbow angles
        'left_elbow': (mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                       mp_pose.PoseLandmark.LEFT_ELBOW.value, 
                       mp_pose.PoseLandmark.LEFT_WRIST.value),
        
        'right_elbow': (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                        mp_pose.PoseLandmark.RIGHT_ELBOW.value, 
                        mp_pose.PoseLandmark.RIGHT_WRIST.value),
        
        # Knee angles
        'left_knee': (mp_pose.PoseLandmark.LEFT_HIP.value, 
                      mp_pose.PoseLandmark.LEFT_KNEE.value, 
                      mp_pose.PoseLandmark.LEFT_ANKLE.value),
        
        'right_knee': (mp_pose.PoseLandmark.RIGHT_HIP.value, 
                       mp_pose.PoseLandmark.RIGHT_KNEE.value, 
                       mp_pose.PoseLandmark.RIGHT_ANKLE.value),
        
        # Hip angles
        'left_hip': (mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                     mp_pose.PoseLandmark.LEFT_HIP.value, 
                     mp_pose.PoseLandmark.LEFT_KNEE.value),
        
        'right_hip': (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                      mp_pose.PoseLandmark.RIGHT_HIP.value, 
                      mp_pose.PoseLandmark.RIGHT_KNEE.value),
        
        # Shoulder angles
        'left_shoulder': (mp_pose.PoseLandmark.LEFT_ELBOW.value, 
                          mp_pose.PoseLandmark.LEFT_SHOULDER.value, 
                          mp_pose.PoseLandmark.LEFT_HIP.value),
        
        'right_shoulder': (mp_pose.PoseLandmark.RIGHT_ELBOW.value, 
                           mp_pose.PoseLandmark.RIGHT_SHOULDER.value, 
                           mp_pose.PoseLandmark.RIGHT_HIP.value),
        
        # Ankle angles
        'left_ankle': (mp_pose.PoseLandmark.LEFT_KNEE.value, 
                       mp_pose.PoseLandmark.LEFT_ANKLE.value, 
                       mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value),  # Foot index or heel
        
        'right_ankle': (mp_pose.PoseLandmark.RIGHT_KNEE.value, 
                        mp_pose.PoseLandmark.RIGHT_ANKLE.value, 
                        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value),
        
        # Wrist angles (Elbow-Wrist-Hand)
        'left_wrist': (mp_pose.PoseLandmark.LEFT_ELBOW.value, 
                       mp_pose.PoseLandmark.LEFT_WRIST.value, 
                       mp_pose.PoseLandmark.LEFT_INDEX.value),  # Index finger or other hand landmark
        
        'right_wrist': (mp_pose.PoseLandmark.RIGHT_ELBOW.value, 
                        mp_pose.PoseLandmark.RIGHT_WRIST.value, 
                        mp_pose.PoseLandmark.RIGHT_INDEX.value),
        
        # Foot Index angles (Ankle-Foot-Toes)
        'left_foot_index': (mp_pose.PoseLandmark.LEFT_ANKLE.value, 
                            mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value, 
                            mp_pose.PoseLandmark.LEFT_HEEL.value),
        
        'right_foot_index': (mp_pose.PoseLandmark.RIGHT_ANKLE.value, 
                             mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value, 
                             mp_pose.PoseLandmark.RIGHT_HEEL.value)
    }

    angles = {}
    
    # Loop through all defined angles and calculate the corresponding angle if visible
    for angle_name, points in angle_definitions.items():
        if check_visibility(landmarks, points):
            angles[angle_name] = calculate_joint_angle(landmarks, *points)
    
    return angles

# Compare user angles to reference angles
def compare_angles(user_angles, reference_angles, tolerance=15):
    feedback = {}
    for joint, user_angle in user_angles.items():
        ref_angle = reference_angles.get(joint, None)
        if ref_angle:
            if abs(user_angle - ref_angle) <= tolerance:
                feedback[joint] = [1, (user_angle - ref_angle)]
            else:
                feedback[joint] = [0, (user_angle - ref_angle)]
    return feedback

def provide_feedback(angle_feedback):
    messages = []

    corrections = {
        'left_elbow': "Adjust your left elbow position.",
        'right_elbow': "Adjust your right elbow position.",
        'left_knee': "Adjust your left knee alignment.",
        'right_knee': "Adjust your right knee alignment.",
        'left_hip': "Check your left hip posture.",
        'right_hip': "Check your right hip posture.",
        'left_shoulder': "Fix your left shoulder positioning.",
        'right_shoulder': "Fix your right shoulder positioning.",
        'left_ankle': "Your left ankle position needs correction.",
        'right_ankle': "Your right ankle position needs correction.",
        'left_wrist': "Adjust your left wrist.",
        'right_wrist': "Adjust your right wrist.",
        'left_foot_index': "Adjust your left foot position.",
        'right_foot_index': "Adjust your right foot position."
    }

    for joint, feedback in angle_feedback.items():
        if feedback[0] == 0: 
            messages.append(f"{corrections[joint]}")
            # messages.append(f"{corrections[joint]} (Angle: {feedback[1]:.0f}°)")

    if messages:
        feedback_message = messages
    else:
        feedback_message = "Great job! All angles are correct."

    return feedback_message


# Example usage:
angle_feedback = {
    'left_elbow': [1, 3.418], 'right_elbow': [1, -4.924], 'left_knee': [0, 34.447],
    'right_knee': [0, 43.328], 'left_hip': [0, 46.419], 'right_hip': [0, 58.134],
    'left_shoulder': [0, -51.328], 'right_shoulder': [0, -63.887], 'left_ankle': [0, 30.281],
    'right_ankle': [0, 35.698], 'left_wrist': [1, 6.354], 'right_wrist': [1, 1.365],
    'left_foot_index': [1, -4.877]
}

# Generate the feedback message
feedback_message = provide_feedback(angle_feedback)

# Print feedback to the user
print(feedback_message)

# Example reference angles for a pose (should be defined based on ideal pose)
reference_angles = {"left_elbow": 156.8494378860644, "left_knee": 91.00880518887023, "left_hip": 91.19988475322886, "right_hip": 86.10251493991747, "left_shoulder": 142.6871778476201, "left_ankle": 88.78916680593252, "left_wrist": 149.4123144162733, "left_foot_index": 14.791581377737295}
feedback_interval = 5
last_feedback_time = time.time()

# Main loop for webcam feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading the webcam feed")
        break

    # Convert the frame to RGB for MediaPipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to find pose landmarks
    results = pose.process(image)

    # Revert the image back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # If pose landmarks are detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Draw landmarks on the image
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get the current time
        current_time = time.time()

        # Only provide feedback every 5 seconds
        if current_time - last_feedback_time >= feedback_interval:
            # Extract angles from the landmarks
            user_angles = extract_angles(landmarks)

            # Compare with reference angles and provide feedback
            feedback = compare_angles(user_angles, reference_angles)
            messages = provide_feedback(feedback)
            print(messages)

            # Update the last feedback time
            last_feedback_time = current_time

    # Show the image in a window
    cv2.imshow('Asana Pose Check', image)

    # Break the loop on pressing 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()