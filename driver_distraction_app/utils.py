import cv2
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

# Model configuration
CLASSES_LIST = [
    "Billboards Viewing", 
    "Normal Driving",
    "Passenger Interaction",
    "Rubbernecking", 
    "Using Electronics Items", 
    "Wildlife Scenery"
]
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
SEQUENCE_LENGTH = 20

def load_tflite_model(model_path):
    """Load and initialize TFLite model"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        logger.info("Model loaded successfully")
        return interpreter
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

def preprocess_frame(frame):
    """Process individual frame"""
    try:
        frame = cv2.resize(frame, (IMAGE_WIDTH, int(IMAGE_HEIGHT*1.5)))
        frame = frame[:IMAGE_HEIGHT, :]  # Crop to square
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame / 255.0  # Normalize
    except Exception as e:
        logger.error(f"Frame processing failed: {str(e)}")
        return None

def extract_frames(video_path):
    """Extract evenly spaced frames"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < SEQUENCE_LENGTH:
            raise ValueError(f"Video too short. Needs ≥{SEQUENCE_LENGTH} frames")
            
        skip_window = max(total_frames // SEQUENCE_LENGTH, 1)
        frames = []
        
        for i in range(SEQUENCE_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_window)
            ret, frame = cap.read()
            if not ret:
                break
                
            processed = preprocess_frame(frame)
            if processed is not None:
                frames.append(processed)
                
        cap.release()
        return np.array(frames, dtype=np.float32) if len(frames) == SEQUENCE_LENGTH else None
        
    except Exception as e:
        logger.error(f"Frame extraction failed: {str(e)}")
        return None

def predict_video(interpreter, video_path):
    """Make predictions on video"""
    try:
        # 1. Extract frames
        frames = extract_frames(video_path)
        if frames is None:
            return None
            
        # 2. Prepare input tensor
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_data = np.expand_dims(frames, axis=0).astype(np.float32)
        
        # 3. Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 4. Process output
        output_details = interpreter.get_output_details()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # 5. Format results
        results = []
        for i, class_name in enumerate(CLASSES_LIST):
            results.append({
                'class': class_name,
                'confidence': float(predictions[i] * 100)
            })
            
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'predicted_class': results[0]['class'],
            'confidence': results[0]['confidence'],
            'all_predictions': results
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return None