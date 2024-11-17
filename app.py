import streamlit as st
import cv2
import torch
import pathlib
import tempfile
import sys
import numpy as np
import os

# pathlib.PosixPath = pathlib.WindowsPath

# Import YOLOv5 model and utilities
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load the custom YOLOv5 model
model_path = 'models/best.pt'
device = select_device('')  # Use CUDA if available
model = DetectMultiBackend(model_path, device=device, dnn=False)
img_size = 640  # Set input size to 640x640 for model

# CSS for styling
st.markdown("""
    <style>
        /* Button styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            font-weight: bold;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }

        /* Progress bar styling */
        .stProgress .st-bs {
            background-color: #3a3f5c !important;
        }
    </style>
""", unsafe_allow_html=True)




# Title with modified color
st.markdown("""
    <h1 style='color: darkblue;'>
        🎾🎾Tennis Match Player and Ball Detection Application using Yolov5
    </h1>
""", unsafe_allow_html=True)

st.write("Upload a video to detect players and balls using the YOLOv5 model.")

# File uploader for video input
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_video:
    # Create a temporary file to store the uploaded video
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path.write(uploaded_video.read())
    temp_video_path.close()

    # Process button
    if st.button("Process"):
        # Load video and initialize parameters
        cap = cv2.VideoCapture(temp_video_path.name)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_path, fourcc, fps, (img_size, img_size))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.write("Processing video...")

        # Progress bar
        progress_bar = st.progress(0)

        # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection model
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                results = model(frame)
        else:
            results = model(frame)

        # Access detection results
        detections = results.xyxy[0].cpu().numpy()  # Assuming detections format [x1, y1, x2, y2, conf, class]

        # Draw bounding boxes with colors based on class
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Set color based on class (assuming class 0 is 'player' and class 1 is 'ball')
            if cls == 0:  # Player
                color = (139, 0, 0)  # Dark blue in BGR
            elif cls == 1:  # Ball
                color = (0, 255, 255)  # Yellow in BGR

            # Draw rectangle and add label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
            label = f"{int(cls)}: {conf:.2f}"  # Replace with actual label text if available
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)

        # Convert BGR to RGB for display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Write frame to output video file
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Display the frame in Streamlit
        stframe.image(frame, channels='RGB', use_container_width=True)

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Ensure consistent frame rate in display
        time.sleep(1 / fps)

    # Release video resources
    cap.release()
    out.release()

        st.success("Detection complete! 🎉🎾🎾🎉")

        # Display processed video
        st.video(output_path)

        # Provide download button
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Processed Video",
                data=file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
      
    
