import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# Import the detector class from the detector.py file instead of Untitled
from detector import CombinedYOLODetector

def main():
    st.title("Combined YOLO Detector App")
    st.sidebar.header("Settings")
    
    # Define available models (these should match the model files in your directory)
    available_models = {
        "KITTI": "KITTI.pt",
        "Light Detection": "light.pt",
        "Pothole Detection": "pothole.pt",
        "Sign Detection": "sign.pt"
    }
    
    # Model selection
    st.sidebar.subheader("Select Models")
    selected_models = {}
    for name, path in available_models.items():
        selected_models[name] = st.sidebar.checkbox(name, value=True if name != "Sign Detection" else False)
    
    # Parameter settings
    st.sidebar.subheader("Detection Parameters")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.0, 1.0, 0.45, 0.05)
    
    # Display settings
    st.sidebar.subheader("Display Settings")
    show_live = st.sidebar.checkbox("Show Live Preview", value=True)
    save_video = st.sidebar.checkbox("Save Output Video", value=True)
    
    # File uploader
    st.subheader("Upload Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    # Camera option
    use_camera = st.checkbox("Use Webcam Instead", value=False)
    camera_index = 0
    if use_camera:
        camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)
    
    # Process button
    process_button = st.button("Process Video")
    
    # Results area
    results_area = st.empty()
    video_output = st.empty()
    stats_area = st.empty()
    
    if process_button:
        # Get selected model paths
        model_paths = [path for name, path in available_models.items() if selected_models[name]]
        
        if not model_paths:
            st.error("Please select at least one model.")
            return
        
        # Initialize the detector
        with st.spinner("Loading models..."):
            detector = CombinedYOLODetector(model_paths)
        
        # Set up input path
        input_path = None
        if use_camera:
            input_path = camera_index
        elif uploaded_file is not None:
            # Create a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            input_path = tfile.name
        else:
            st.error("Please upload a video file or select webcam option.")
            return
        
        # Set up output path
        output_path = None
        if save_video:
            if uploaded_file is not None:
                # Create output path from uploaded filename
                file_name = uploaded_file.name
                output_path = f"output_{file_name}"
            else:
                # Create output path for webcam
                output_path = f"output_webcam_{camera_index}.mp4"
        
        # Process video
        results_area.text("Processing video... This may take some time.")
        try:
            stats = detector.process_video(
                input_path=input_path,
                output_path=output_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                show_live=show_live,
                save_video=save_video
            )
            
            # Display stats
            results_area.success("Processing completed!")
            stats_area.subheader("Detection Statistics")
            
            # Display statistics in a more readable format
            col1, col2 = stats_area.columns(2)
            with col1:
                st.write("Detections by Class:")
                for cls, count in stats['detections_by_class'].items():
                    st.write(f"- {cls}: {count}")
            
            with col2:
                st.write("Detections by Model:")
                for model, count in stats['detections_by_model'].items():
                    st.write(f"- {model}: {count}")
            
            st.write(f"Total Frames Processed: {stats['total_frames']}")
            
            # Display output video if saved
            if save_video and output_path and os.path.exists(output_path):
                video_output.subheader("Output Video")
                video_output.video(output_path)
        
        except Exception as e:
            results_area.error(f"Error during processing: {e}")
        
        # Clean up temporary file if created
        if not use_camera and uploaded_file is not None:
            os.unlink(tfile.name)

if __name__ == "__main__":
    main()