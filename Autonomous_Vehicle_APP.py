import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from pathlib import Path

# Import the detector class from the detector.py file
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
    
    # Check if we're in a headless environment (like Streamlit Cloud)
    # by trying to get DISPLAY environment variable
    import os
    has_display = os.environ.get('DISPLAY', '') != ''
    
    # Disable live preview option if in a headless environment
    if not has_display:
        show_live = False
        st.sidebar.warning("Live preview not available in headless environment (like Streamlit Cloud)")
        st.sidebar.checkbox("Show Live Preview", value=False, disabled=True)
    else:
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
    progress_bar = st.empty()
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
            try:
                detector = CombinedYOLODetector(model_paths)
                results_area.success("Models loaded successfully!")
            except Exception as e:
                results_area.error(f"Error loading models: {e}")
                return
        
        # Set up input path
        input_path = None
        temp_file = None
        
        try:
            if use_camera:
                input_path = camera_index
                results_area.info(f"Using camera with index {camera_index}")
            elif uploaded_file is not None:
                # Create a temporary file
                results_area.info("Processing uploaded video...")
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_file.write(uploaded_file.read())
                input_path = temp_file.name
                results_area.info(f"Video saved to temporary file: {input_path}")
            else:
                results_area.error("Please upload a video file or select webcam option.")
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
                
                results_area.info(f"Output will be saved to: {output_path}")
            
            # Add a custom processing method to show progress in Streamlit
            def process_with_progress(detector, input_path, output_path, conf_threshold, iou_threshold, show_live, save_video):
                # Open video capture
                if isinstance(input_path, int) or (isinstance(input_path, str) and input_path.isdigit()):
                    cap = cv2.VideoCapture(int(input_path))
                    progress_bar.text("Opened camera feed - press 'q' in the preview window to stop")
                else:
                    cap = cv2.VideoCapture(input_path)
                    progress_bar.text(f"Opened video file: {input_path}")
                
                if not cap.isOpened():
                    progress_bar.error(f"Error: Could not open video source {input_path}")
                    return {}
                
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # For webcam, we don't know the total frames
                if isinstance(input_path, int) or (isinstance(input_path, str) and input_path.isdigit()):
                    total_frames = 0  # Unknown for webcam
                
                # Initialize video writer if saving
                writer = None
                if save_video and output_path:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    progress_bar.text(f"Saving output to: {output_path}")
                
                # Stats for summary
                frame_count = 0
                detection_stats = defaultdict(int)
                model_stats = defaultdict(int)
                
                # Create a progress bar if total frames is known
                progress = None
                if total_frames > 0:
                    progress = progress_bar.progress(0)
                else:
                    progress_bar.text("Processing frames... (Press 'q' in preview window to stop)")
                
                start_time = time.time()
                processed_frames = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    processed_frames += 1
                    
                    # Update progress every 5 frames or so
                    if processed_frames % 5 == 0:
                        elapsed_time = time.time() - start_time
                        fps_processing = processed_frames / elapsed_time if elapsed_time > 0 else 0
                        
                        if total_frames > 0:
                            progress.progress(min(frame_count / total_frames, 1.0))
                            progress_bar.text(f"Processing frame {frame_count}/{total_frames} - {fps_processing:.2f} FPS")
                        else:
                            progress_bar.text(f"Processed {frame_count} frames - {fps_processing:.2f} FPS")
                    
                    # Process frame with detector
                    processed_frame, results = detector.detect_on_frame(frame, conf_threshold, iou_threshold)
                    
                    # Update statistics
                    for cls_name, detections in results.items():
                        detection_stats[cls_name] += len(detections)
                        for det in detections:
                            model_stats[det['model']] += 1
                    
                    # Display frame if requested and not in headless environment
                    if show_live:
                        try:
                            cv2.imshow('Combined YOLO Detection', processed_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        except:
                            # If imshow fails, we're probably in a headless environment
                            # Just continue without displaying
                            show_live = False  # Disable for future frames
                    
                    # Save frame if requested
                    if writer is not None:
                        writer.write(processed_frame)
                
                # Cleanup
                cap.release()
                if writer is not None:
                    writer.release()
                if show_live:
                    try:
                        cv2.destroyAllWindows()
                    except:
                        pass  # Ignore errors when closing windows in headless environment
                
                # Prepare and return statistics
                stats = {
                    'total_frames': frame_count,
                    'detections_by_class': dict(detection_stats),
                    'detections_by_model': dict(model_stats),
                    'processing_time': time.time() - start_time,
                    'average_fps': frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                }
                
                return stats
            
            # Import defaultdict for our custom processing function
            from collections import defaultdict
            
            # Process video
            results_area.text("Processing video... This may take some time.")
            
            try:
                stats = process_with_progress(
                    detector=detector,
                    input_path=input_path,
                    output_path=output_path,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    show_live=show_live,
                    save_video=save_video
                )
                
                # Display stats
                results_area.success("Processing completed!")
                
                # Display statistics in a more readable format
                stats_area.subheader("Detection Statistics")
                
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
                st.write(f"Processing Time: {stats['processing_time']:.2f} seconds")
                st.write(f"Average FPS: {stats['average_fps']:.2f}")
                
                # Display output video if saved
                if save_video and output_path and os.path.exists(output_path):
                    video_output.subheader("Output Video")
                    video_output.video(output_path)
            
            except Exception as e:
                results_area.error(f"Error during processing: {str(e)}")
                st.exception(e)  # This will show the full traceback
        
        finally:
            # Clean up temporary file if created
            if temp_file is not None:
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                    results_area.info(f"Temporary file {temp_file.name} has been cleaned up.")
                except:
                    results_area.warning(f"Could not delete temporary file {temp_file.name}.")

if __name__ == "__main__":
    main()
