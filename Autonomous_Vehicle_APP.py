import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os
import time
from PIL import Image
import io
import base64
from datetime import datetime

class CombinedYOLODetector:
    def __init__(self, model_paths):
        """
        Initialize the combined YOLO detector with multiple model paths
        
        Args:
            model_paths (list): List of paths to YOLOv8 .pt model files
        """
        self.models = []
        self.model_names = []
        
        # Model name mapping (file to display name)
        model_name_mapping = {
            "KITTI.pt": "KITTI",
            "light.pt": "Light Detection",
            "pothole.pt": "Pothole Detection",
            "sign.pt": "Sign Detection"
        }
        
        # Load all models
        for model_path in model_paths:
            try:
                model = YOLO(model_path)
                
                # Get the base filename
                base_filename = os.path.basename(model_path)
                
                # Use the friendly name if available, otherwise use the file name without extension
                if base_filename in model_name_mapping:
                    model_name = model_name_mapping[base_filename]
                else:
                    model_name = base_filename.split('.')[0]
                    
                self.models.append(model)
                self.model_names.append(model_name)
                st.success(f"Loaded model: {model_name}")
            except Exception as e:
                st.error(f"Failed to load model {model_path}: {e}")
    
    def detect_on_frame(self, frame, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run all models on a single frame and combine detections
        
        Args:
            frame (numpy.ndarray): Input frame
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            
        Returns:
            tuple: (frame with all detections drawn, combined results dictionary)
        """
        if not self.models:
            st.error("No models loaded. Please check model paths.")
            return frame, {}
        
        combined_results = defaultdict(list)
        drawn_frame = frame.copy()
        
        # Process frame with each model
        for i, model in enumerate(self.models):
            model_name = self.model_names[i]
            results = model(frame, conf=conf_threshold, iou=iou_threshold)[0]
            
            # Extract detections
            boxes = results.boxes.cpu().numpy()
            
            if len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = results.names[cls]
                    
                    # Store results by class name
                    combined_results[cls_name].append({
                        'box': (x1, y1, x2, y2),
                        'conf': conf,
                        'model': model_name
                    })
                    
                    # Draw bounding box on the frame
                    color = self._get_color(model_name)
                    cv2.rectangle(drawn_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with class, confidence, and model name
                    label = f"{cls_name} ({conf:.2f}) - {model_name}"
                    cv2.putText(drawn_frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return drawn_frame, combined_results
    
    def _get_color(self, model_name):
        """
        Generate a consistent color based on model name
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            tuple: BGR color values
        """
        # Generate a hash from the model name to get consistent colors
        hash_val = hash(model_name) % 360
        
        # Convert HSV to BGR for OpenCV
        h = hash_val / 360.0
        s = v = 0.9
        
        # HSV to RGB conversion
        hi = int(h * 6) % 6
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        r, g, b = 0, 0, 0
        if hi == 0: r, g, b = v, t, p
        elif hi == 1: r, g, b = q, v, p
        elif hi == 2: r, g, b = p, v, t
        elif hi == 3: r, g, b = p, q, v
        elif hi == 4: r, g, b = t, p, v
        elif hi == 5: r, g, b = v, p, q
        
        # Convert to 0-255 range for BGR
        return (int(b * 255), int(g * 255), int(r * 255))

    def process_image(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        Process a single image with all models
        
        Args:
            image (numpy.ndarray): Input image
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            
        Returns:
            tuple: (processed image in RGB, results dictionary)
        """
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
            # Check if we need to convert from RGB to BGR
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        # Process the image with detections
        processed_image, results = self.detect_on_frame(image, conf_threshold, iou_threshold)
        
        # Convert back to RGB for Streamlit display
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        return processed_image_rgb, results

    def process_video(self, input_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Process a video file with all models for Streamlit
        
        Args:
            input_path (str): Path to input video file
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            
        Returns:
            tuple: (Statistics about detections, path to output video file)
        """
        # Open video capture
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            st.error(f"Error: Could not open video source {input_path}")
            return {}, None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a temporary file for the output video
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Stats for summary
        frame_count = 0
        detection_stats = defaultdict(int)
        model_stats = defaultdict(int)
        
        # Create a placeholder for the video frames
        frame_placeholder = st.empty()
        
        # Create placeholders for the stats
        stats_col1, stats_col2 = st.columns(2)
        class_stats_placeholder = stats_col1.empty()
        model_stats_placeholder = stats_col2.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame with all models
            processed_frame, results = self.detect_on_frame(frame, conf_threshold, iou_threshold)
            
            # Convert to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Write the BGR frame to the output video file
            out.write(processed_frame)
            
            # Update statistics
            for cls_name, detections in results.items():
                detection_stats[cls_name] += len(detections)
                for det in detections:
                    model_stats[det['model']] += 1
            
            # Display the frame in Streamlit
            frame_placeholder.image(processed_frame_rgb, caption='Processed Frame', use_container_width=True)
            
            # Update progress bar and stats display
            progress = min(frame_count / total_frames, 1.0) if total_frames > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Update statistics display
            class_stats_placeholder.write("### Detections by Class")
            class_stats_placeholder.write(dict(detection_stats))
            
            model_stats_placeholder.write("### Detections by Model")
            model_stats_placeholder.write(dict(model_stats))
            
            # Add a small delay to make the video playback smoother in Streamlit
            time.sleep(0.01)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final statistics
        stats = {
            'total_frames': frame_count,
            'detections_by_class': dict(detection_stats),
            'detections_by_model': dict(model_stats)
        }
        
        status_text.text(f"Completed processing {frame_count} frames")
        
        return stats, output_path

    def process_webcam(self, conf_threshold=0.25, iou_threshold=0.45):
        """
        Process webcam feed with all models for Streamlit
        
        Args:
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            
        Returns:
            str: Path to the saved video file
        """
        # Try to open the webcam with different indices
        cap = None
        camera_index = 0
        max_attempts = 3
        
        for i in range(max_attempts):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_index = i
                st.success(f"Successfully opened camera at index {i}")
                break
            else:
                cap.release()
        
        if cap is None or not cap.isOpened():
            st.error("Error: Could not open any camera. Please check your camera connection.")
            return None
        
        # Create a temporary file for the output video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'_webcam_{timestamp}.mp4').name
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20  # Fixed fps for webcam recording
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create placeholders
        frame_placeholder = st.empty()
        stats_col1, stats_col2 = st.columns(2)
        class_stats_placeholder = stats_col1.empty()
        model_stats_placeholder = stats_col2.empty()
        
        # Stats for summary
        detection_stats = defaultdict(int)
        model_stats = defaultdict(int)
        frame_count = 0
        
        # Create a stop button
        stop_button_placeholder = st.empty()
        stop_button = stop_button_placeholder.button("Stop Webcam")
        
        # Start recording
        recording = True
        st.info(f"Recording from camera index {camera_index}. Press 'Stop Webcam' to finish.")
        
        while cap.isOpened() and recording and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame with all models
            processed_frame, results = self.detect_on_frame(frame, conf_threshold, iou_threshold)
            
            # Convert to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Write the BGR frame to the output video file
            out.write(processed_frame)
            
            # Update statistics (only count the current frame)
            frame_detection_stats = defaultdict(int)
            frame_model_stats = defaultdict(int)
            
            for cls_name, detections in results.items():
                frame_detection_stats[cls_name] += len(detections)
                detection_stats[cls_name] += len(detections)
                for det in detections:
                    frame_model_stats[det['model']] += 1
                    model_stats[det['model']] += 1
            
            # Display the frame in Streamlit
            frame_placeholder.image(processed_frame_rgb, caption='Live Detection', use_container_width=True)
            
            # Update statistics display
            class_stats_placeholder.write("### Current Frame Detections by Class")
            class_stats_placeholder.write(dict(frame_detection_stats))
            
            model_stats_placeholder.write("### Current Frame Detections by Model")
            model_stats_placeholder.write(dict(frame_model_stats))
            
            # Check if the stop button was pressed
            stop_button = stop_button_placeholder.button("Stop Webcam", key=f"stop_webcam_{frame_count}")
            if stop_button:
                recording = False
                break
                
            # Add a small delay to reduce CPU usage
            time.sleep(0.01)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Show aggregate statistics
        st.write("### Total Detections by Class")
        st.write(dict(detection_stats))
        
        st.write("### Total Detections by Model")
        st.write(dict(model_stats))
        
        st.success(f"Recorded {frame_count} frames to video file")
        
        return output_path


def get_image_download_link(img, filename, text):
    """
    Generate a download link for an image
    """
    # Convert to PIL Image if it's numpy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        
    # Save to bytes
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    
    # Generate download link
    btn = f'<a href="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" download="{filename}" style="display: inline-block; padding: 0.25em 0.5em; text-decoration: none; color: white; background: #4CAF50; border-radius: 3px; border: none; cursor: pointer; font-size: 16px;">{text}</a>'
    return btn


def get_binary_file_downloader_html(file_path, file_label):
    """
    Generate a download link for a binary file
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:video/mp4;base64,{b64}" download="{os.path.basename(file_path)}" style="display: inline-block; padding: 0.25em 0.5em; text-decoration: none; color: white; background: #2196F3; border-radius: 3px; border: none; cursor: pointer; font-size: 16px;">{file_label}</a>'


def main():
    st.set_page_config(page_title="Traffic Object Detection System", layout="wide")
    
    st.title("Traffic Object Detection System")
    st.markdown("Detect traffic-related objects using specialized YOLO models")
    
    # Sidebar for configurations
    st.sidebar.header("Configuration")
    
    # Predefined models
    available_models = {
        "KITTI": "KITTI.pt",
        "Light Detection": "light.pt",
        "Pothole Detection": "pothole.pt",
        "Sign Detection": "sign.pt"
    }
    
    # Model selection section
    st.sidebar.subheader("Select Models")
    selected_models = {}
    for model_name, model_file in available_models.items():
        selected_models[model_name] = st.sidebar.checkbox(f"Use {model_name}", value=True)
    
    # Detection parameters
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.45, 0.05)
    
    # Initialize detector with selected models
    model_paths = []
    
    # Check if any model is selected
    if any(selected_models.values()):
        # Get the paths for selected models
        for model_name, is_selected in selected_models.items():
            if is_selected:
                model_file = available_models[model_name]
                model_paths.append(model_file)
        
        # Initialize the detector with the model paths
        detector = CombinedYOLODetector(model_paths)
        
        # Input type selection
        input_type = st.radio(
            "Select Input Type", 
            ["Image", "Video", "Webcam"]
        )
        
        if input_type == "Image":
            # Image upload and processing
            uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            
            if uploaded_image is not None:
                # Read and display the original image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Process button
                if st.button("Detect Objects"):
                    with st.spinner("Processing image..."):
                        # Convert PIL Image to numpy array
                        image_np = np.array(image)
                        
                        # Process the image
                        processed_image, results = detector.process_image(image_np, conf_threshold, iou_threshold)
                        
                        # Display the processed image
                        st.image(processed_image, caption="Processed Image", use_container_width=True)
                        
                        # Generate a timestamp for the filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"detected_image_{timestamp}.png"
                        
                        # Create download button
                        st.markdown(get_image_download_link(processed_image, filename, "Download Processed Image"), unsafe_allow_html=True)
                        
                        # Display results
                        st.subheader("Detection Results")
                        
                        # Create expandable sections for each class
                        for cls_name, detections in results.items():
                            with st.expander(f"{cls_name} ({len(detections)} detections)"):
                                for i, det in enumerate(detections):
                                    st.write(f"Detection {i+1}: Confidence = {det['conf']:.2f}, Model = {det['model']}")
        
        elif input_type == "Video":
            # Video upload and processing
            uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
            
            if uploaded_video is not None:
                # Save the uploaded video to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    temp_file.write(uploaded_video.read())
                    video_path = temp_file.name
                
                # Process button
                if st.button("Process Video"):
                    try:
                        # Process the video
                        with st.spinner("Processing video... This may take some time depending on the video length."):
                            stats, output_video_path = detector.process_video(video_path, conf_threshold, iou_threshold)
                        
                        # Display final statistics
                        st.subheader("Detection Statistics")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### Detections by Class")
                            st.write(stats['detections_by_class'])
                        
                        with col2:
                            st.write("### Detections by Model")
                            st.write(stats['detections_by_model'])
                            
                        st.write(f"Total Frames Processed: {stats['total_frames']}")
                        
                        # Create download button for the processed video
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        download_filename = f"processed_video_{timestamp}.mp4"
                        st.markdown(get_binary_file_downloader_html(output_video_path, "Download Processed Video"), unsafe_allow_html=True)
                        
                    finally:
                        # Clean up the temporary file
                        os.unlink(video_path)
        
        elif input_type == "Webcam":
            # Webcam processing
            if st.button("Start Webcam"):
                # Process webcam feed
                output_video_path = detector.process_webcam(conf_threshold, iou_threshold)
                
                if output_video_path:
                    # Create download button for the recorded video
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_filename = f"webcam_recording_{timestamp}.mp4"
                    st.markdown(get_binary_file_downloader_html(output_video_path, "Download Webcam Recording"), unsafe_allow_html=True)
    else:
        st.warning("Please select at least one model from the sidebar to begin.")
        
    # Add instructions and info at the bottom
    with st.expander("Instructions & Information"):
        st.markdown("""
        ### How to use this app:
        
        1. **Select Models**: Choose which detection models to use from the sidebar.
        2. **Configure Settings**: Adjust the confidence and IOU thresholds as needed.
        3. **Select Input Type**: Choose between image, video, or webcam input.
        4. **Process Content**: Upload an image/video or start the webcam, then run detection.
        5. **Download Results**: After processing, you can download the processed image or video.
        
        ### Available Models:
        - **KITTI**: General traffic object detection based on the KITTI dataset
        - **Light Detection**: Traffic light detection with state classification
        - **Pothole Detection**: Identifies potholes and road damage
        - **Sign Detection**: Detects and classifies traffic signs
        
        ### Notes:
        - You can select multiple models to compare their detections side-by-side.
        - Each model uses a different color for its bounding boxes.
        - The app shows both per-class and per-model statistics.
        - For webcam use, the system will automatically detect an available camera.
        - Processed content is provided in a downloadable format.
        """)


if __name__ == "__main__":
    # Check for required model files
    required_models = ["KITTI.pt", "light.pt", "pothole.pt"]
    missing_models = []
    
    for model_file in required_models:
        if not os.path.exists(model_file):
            missing_models.append(model_file)
    
    if missing_models:
        st.error(f"The following model files are missing from the current directory: {', '.join(missing_models)}")
        st.error("Please add the missing model files and restart the application.")
    else:
        main()
