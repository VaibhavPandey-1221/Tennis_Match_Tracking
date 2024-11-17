import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os
import time

# Define model path
model_path = 'best.pt'  # Replace with your actual .pt file path

# Load YOLOv5 model
try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e
# Main App UI
st.title('🎾 Tennis Tracking App')
st.write("Upload a tennis video to detect and track players in real-time.")

# File uploader for video input
uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Set up the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    st.write("⏳ Processing video... Please wait.")

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

    st.success("🎉 Video processing complete!")

    # Provide download button for the processed video
    st.write("📥 Download the processed video:")
    with open(output_video_path, 'rb') as f:
        st.download_button(
            label="⬇ Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)
