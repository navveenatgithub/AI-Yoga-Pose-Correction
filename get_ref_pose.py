import cv2
import mediapipe as mp
import numpy as np
import os
import json
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Middle point (e.g., elbow)
    c = np.array(c)  # Third point (e.g., wrist)

    # Calculate the vectors between the points
    ba = a - b
    bc = c - b

    # Calculate the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    # Convert the angle to degrees
    return np.degrees(angle)

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

# Process folder of images
def analyze_asana_images(folder_path, output_reference_file):
    all_angles = []

    # Loop over all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)

            # Convert the image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform pose detection
            result = pose.process(image_rgb)

            if result.pose_landmarks:
                # Extract angles for this pose
                angles = extract_angles(result.pose_landmarks.landmark)
                all_angles.append(angles)

                # Optionally, draw pose landmarks on the image (for visualization)
                # annotated_image = image.copy()
                # mp_drawing.draw_landmarks(
                #     annotated_image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # cv2.imshow("Pose", annotated_image)
                # cv2.waitKey(0)

    # Average angles for reference
    avg_angles = {}
    if all_angles:
        keys = all_angles[0].keys()
        for key in keys:
            avg_angles[key] = np.mean([angles[key] for angles in all_angles if key in angles])

    # Save the average angles to a reference file (e.g., JSON)
    with open(output_reference_file, 'w') as f:
        json.dump(avg_angles, f)

    print(f"Reference angles saved to {output_reference_file}")

# Main function
if __name__ == "__main__":
    folder_path = r"E:\SYM - Pose Detection\dataset\archive\dataset\utkatasana"
    output_reference_file = "utkatasana.json"  # Output file for reference data

    analyze_asana_images(folder_path, output_reference_file)
