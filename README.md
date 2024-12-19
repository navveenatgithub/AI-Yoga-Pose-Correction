


The AI-based smart yoga mat application is a next-generation fitness tool designed to assist users in performing yoga poses correctly. By leveraging machine learning, computer vision, and real-time pose analysis, the application provides instant feedback and corrections to users during their yoga sessions. It compares the user's body posture with pre-trained pose data, identifies deviations, and guides them through voice-assisted corrections, ensuring proper alignment and reducing the risk of injury.

TECH STACK :

Programming Language: Python

Libraries and Frameworks:
TensorFlow/Keras: For pose classification model training
MediaPipe: For real-time pose detection and keypoint extraction
NumPy: For angle calculations
OpenCV: For real-time video feed processing
pyttsx3: For text-to-speech voice assistance
Matplotlib: For visualizing pose data during development

Tools:
Anaconda
Spyder IDE or Jupyter Notebook
Pre-trained datasets for yoga pose angles

WORKING :

Pose Data Collection:
Pre-trained dataset images are analyzed for key body points.
Joint angles are calculated and stored as reference values.
Real-Time Pose Detection:
The user's live video feed is captured and processed using MediaPipe to detect keypoints.
Angles between joints are calculated for comparison.
Pose Correction:
The real-time angles are compared with reference pose angles.
If deviations are detected, corrective instructions are generated.
Voice-Assisted Feedback:
Textual corrections are converted into voice prompts using pyttsx3.
Users receive real-time guidance to adjust their poses.

KEY FEATURES :

Real-Time Pose Detection: Continuous monitoring of the user's pose during yoga sessions.
Pose Comparison: Accurate comparison with reference data to identify misalignments.
Voice Guidance: Instant verbal feedback for intuitive and hands-free corrections.
User-Friendly Interface: Intuitive and engaging application design for all skill levels.
Customizability: Ability to add new poses or customize pose correction thresholds.
Data Analytics: Tracks user progress over time, offering insights into improvement.
