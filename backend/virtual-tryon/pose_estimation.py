import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

def generate_pose_map(image):
   
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils

    image_np = np.array(image)
    results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    annotated_image = image_np.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    pose_image = Image.fromarray(annotated_image)
    return pose_image
