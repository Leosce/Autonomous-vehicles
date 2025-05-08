import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

class CombinedYOLODetector:
    def __init__(self, model_paths):
        """
        Initialize the combined YOLO detector with multiple model paths
        
        Args:
            model_paths (list): List of paths to YOLOv8 .pt model files
        """
        self.models = []
        self.model_names = []
        
        # Load all models
        for model_path in model_paths:
            try:
                model = YOLO(model_path)
                model_name = model_path.split('/')[-1].split('.')[0]  # Extract name from path
                self.models.append(model)
                self.model_names.append(model_name)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Failed to load model {model_path}: {e}")
    
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
            print("No models loaded. Please check model paths.")
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

    def process_video(self, input_path, output_path=None, conf_threshold=0.25, 
                     iou_threshold=0.45, show_live=True, save_video=True):
        """
        Process a video file with all models
        
        Args:
            input_path (str): Path to input video file or camera index
            output_path (str, optional): Path to save output video
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IOU threshold for NMS
            show_live (bool): Whether to display video during processing
            save_video (bool): Whether to save processed video
            
        Returns:
            dict: Statistics about detections
        """
        # Open video capture
        if isinstance(input_path, int) or input_path.isdigit():
            cap = cv2.VideoCapture(int(input_path))
            print(f"Opened camera {input_path}")
        else:
            cap = cv2.VideoCapture(input_path)
            print(f"Opened video file: {input_path}")
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {input_path}")
            return {}
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize video writer if saving
        writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving output to: {output_path}")
        
        # Stats for summary
        frame_count = 0
        detection_stats = defaultdict(int)
        model_stats = defaultdict(int)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Process frame with all models
            processed_frame, results = self.detect_on_frame(frame, conf_threshold, iou_threshold)
            
            # Update statistics
            for cls_name, detections in results.items():
                detection_stats[cls_name] += len(detections)
                for det in detections:
                    model_stats[det['model']] += 1
            
            # Display frame if requested
            if show_live:
                cv2.imshow('Combined YOLO Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Save frame if requested
            if writer is not None:
                writer.write(processed_frame)
        
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        if show_live:
            cv2.destroyAllWindows()
        
        # Prepare and return statistics
        stats = {
            'total_frames': frame_count,
            'detections_by_class': dict(detection_stats),
            'detections_by_model': dict(model_stats)
        }
        
        print(f"Processed {frame_count} frames")
        print(f"Detections by class: {dict(detection_stats)}")
        print(f"Detections by model: {dict(model_stats)}")
        
        return stats
