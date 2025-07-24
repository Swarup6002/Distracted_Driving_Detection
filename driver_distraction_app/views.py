# views.py
import os
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import tempfile
import logging
from datetime import datetime
from .auth_utils import login_required

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
CLASSES_LIST = [
    "Billboards Viewing", 
    "Normal Driving",
    "Passenger Interaction",
    "Rubbernecking", 
    "Using Electronics Items", 
    "Wildlife Scenery"
]

# Initialize models
try:
    # Load TFLite model for webcam analysis
    TFLITE_MODEL_PATH = os.path.join(settings.BASE_DIR, '99%driver_distraction96by96b.tflite')
    logger.info(f"Loading TFLite model from {TFLITE_MODEL_PATH}")
    
    if not os.path.exists(TFLITE_MODEL_PATH):
        raise FileNotFoundError(f"TFLite model file not found at {TFLITE_MODEL_PATH}")
    
    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    tflite_interpreter.allocate_tensors()
    
    # Get TFLite model details
    tflite_input_details = tflite_interpreter.get_input_details()
    tflite_output_details = tflite_interpreter.get_output_details()
    logger.info(f"TFLite model loaded successfully. Input: {tflite_input_details}, Output: {tflite_output_details}")
    
    # Load H5 model for uploaded videos
    H5_MODEL_PATH = os.path.join(settings.BASE_DIR, 'final_model.h5')
    logger.info(f"Loading H5 model from {H5_MODEL_PATH}")
    lstm_model = load_model(H5_MODEL_PATH)
    
    # Load VGG16 for feature extraction (for uploaded videos)
    logger.info("Loading VGG16 for feature extraction")
    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    vgg_model.trainable = False
    
except Exception as e:
    logger.error(f"Model initialization failed: {str(e)}")
    raise ImportError(f"Failed to initialize models: {str(e)}")

# Settings for different modes
WEB_CAM_SETTINGS = {
    'model': 'tflite',
    'image_size': (96, 96),
    'sequence_length': 20,
    'frame_interval': 0.08,
    'record_seconds': 10
}

UPLOAD_VIDEO_SETTINGS = {
    'model': 'h5',
    'image_size': (224, 224),  # VGG16 input size
    'sequence_length': 20,
    'frame_interval': 0.1,
    'record_seconds': 10
}

def home(request):
    """Render the home page"""
    return render(request, 'index.html')
@login_required
def upload_video(request):
    """Handle video file uploads and processing using H5 model with VGG features"""
    if request.method == 'POST':
        if 'video' not in request.FILES:
            return render(request, 'upload_video.html', {'error': 'No video selected'})
        
        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, 'uploads'))
        filename = None
        
        try:
            video_file = request.FILES['video']
            filename = fs.save(video_file.name, video_file)
            video_path = fs.path(filename)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            features_sequence = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % int(1/UPLOAD_VIDEO_SETTINGS['frame_interval']) == 0:
                    features = extract_vgg_features(frame)
                    if features is not None:
                        features_sequence.append(features)
                
                frame_count += 1
                
                if len(features_sequence) >= UPLOAD_VIDEO_SETTINGS['sequence_length'] * 2:
                    break
            
            cap.release()
            
            if len(features_sequence) < UPLOAD_VIDEO_SETTINGS['sequence_length']:
                raise ValueError(f"Insufficient frames ({len(features_sequence)}/{UPLOAD_VIDEO_SETTINGS['sequence_length']})")
            
            skip = max(1, len(features_sequence) // UPLOAD_VIDEO_SETTINGS['sequence_length'])
            selected_features = features_sequence[::skip][:UPLOAD_VIDEO_SETTINGS['sequence_length']]
            
            if len(selected_features) < UPLOAD_VIDEO_SETTINGS['sequence_length']:
                last_feature = selected_features[-1] if selected_features else np.zeros(512)
                selected_features.extend([last_feature] * (UPLOAD_VIDEO_SETTINGS['sequence_length'] - len(selected_features)))
            
            input_data = np.array([selected_features], dtype=np.float32)
            predictions = lstm_model.predict(input_data, verbose=0)
            
            all_predictions = []
            for cls, conf in zip(CLASSES_LIST, predictions[0]):
                confidence = min(max(float(conf * 100), 0), 100)
                all_predictions.append({
                    'class': cls,
                    'confidence': confidence
                })
            
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return render(request, 'results.html', {
                'video_name': filename,
                'results': {
                    'predicted_class': all_predictions[0]['class'],
                    'confidence': all_predictions[0]['confidence'],
                    'all_predictions': all_predictions
                }
            })
            
        except Exception as e:
            if filename and fs.exists(filename):
                fs.delete(filename)
            logger.error(f"Video processing failed: {str(e)}")
            return render(request, 'upload_video.html', {
                'error': f'Processing failed: {str(e)}'
            })
    
    return render(request, 'upload_video.html')



def extract_vgg_features(frame):
    """Extract features from a single frame using VGG16 (for uploaded videos)"""
    try:
        frame = cv2.resize(frame, UPLOAD_VIDEO_SETTINGS['image_size'])
        frame = preprocess_input(frame.astype(np.float32))
        features = vgg_model.predict(np.expand_dims(frame, axis=0), verbose=0)
        return features[0]
    except Exception as e:
        logger.error(f"Feature extraction failed: {str(e)}")
        return None

@csrf_exempt
def webcam_analysis(request):
    """Optimized webcam processing with direct frame handling using TFLite model"""
    if request.method == 'POST':
        cap = None
        try:
            cap = initialize_builtin_camera()
            if not cap or not cap.isOpened():
                raise ValueError("Could not access webcam")
            
            frames = []
            start_time = time.time()
            
            while (time.time() - start_time) < WEB_CAM_SETTINGS['record_seconds']:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                processed = preprocess_frame(frame, source='webcam')
                if processed is not None:
                    frames.append(processed)
                time.sleep(WEB_CAM_SETTINGS['frame_interval'])
            
            cap.release()
            
            if len(frames) < WEB_CAM_SETTINGS['sequence_length']:
                raise ValueError(f"Insufficient frames ({len(frames)}/{WEB_CAM_SETTINGS['sequence_length']})")
            
            # Match training frame selection
            skip = max(1, len(frames) // WEB_CAM_SETTINGS['sequence_length'])
            selected_frames = frames[::skip][:WEB_CAM_SETTINGS['sequence_length']]
            
            if len(selected_frames) < WEB_CAM_SETTINGS['sequence_length']:
                last_frame = selected_frames[-1] if selected_frames else np.zeros((WEB_CAM_SETTINGS['image_size'][0], WEB_CAM_SETTINGS['image_size'][1], 3))
                selected_frames.extend([last_frame] * (WEB_CAM_SETTINGS['sequence_length'] - len(selected_frames)))
            
            input_data = np.array(selected_frames, dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            
            expected_shape = tuple(tflite_input_details[0]['shape'])
            if input_data.shape != expected_shape:
                raise ValueError(f"Input shape mismatch. Expected {expected_shape}, got {input_data.shape}")
            
            tflite_interpreter.set_tensor(tflite_input_details[0]['index'], input_data)
            tflite_interpreter.invoke()
            predictions = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])
            
            all_predictions = []
            for cls, conf in zip(CLASSES_LIST, predictions[0]):
                confidence = min(max(float(conf * 100), 0), 100)
                all_predictions.append({
                    'class': cls,
                    'confidence': confidence
                })
            
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            if all_predictions[0]['confidence'] < 50:
                raise ValueError(f"Low confidence prediction: {all_predictions[0]['confidence']}%")
            
            return JsonResponse({
                'success': True,
                'predicted_class': all_predictions[0]['class'],
                'confidence': all_predictions[0]['confidence'],
                'all_predictions': all_predictions,
                'processing_time': round(time.time() - start_time, 2)
            })
            
        except Exception as e:
            if cap and cap.isOpened():
                cap.release()
            logger.error(f"Webcam analysis failed: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': str(e),
                'message': 'Webcam analysis failed'
            }, status=400)
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    }, status=405)

def initialize_builtin_camera():
    """Optimized camera initialization with higher resolution for webcam"""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cap = None
    
    for backend in backends:
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                # Set higher resolution for webcam
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

                # Optimized settings for webcam
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cap.set(cv2.CAP_PROP_FOCUS, 50)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, -4)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Warm-up camera
                for _ in range(10):
                    cap.read()
                    time.sleep(0.1)
                break
        except:
            if cap:
                cap.release()
            continue
    
    return cap

def preprocess_frame(frame, source='webcam'):
    """Preprocess frame according to the source (webcam or video)"""
    try:
        if source == 'webcam':
            # For webcam - use TFLite model preprocessing
            frame = cv2.resize(frame, WEB_CAM_SETTINGS['image_size'])
        else:
            # For uploaded video - VGG16 preprocessing (handled in extract_vgg_features)
            frame = cv2.resize(frame, UPLOAD_VIDEO_SETTINGS['image_size'])
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.float32) / 255.0
    except Exception as e:
        logger.error(f"Frame preprocessing error: {str(e)}")
        return None

def live_analysis_view(request):
    """Render the webcam analysis interface"""
    return render(request, 'webcam_video.html')

@csrf_exempt
def process_webcam_frame(request):
    """Process individual frame from webcam using TFLite model"""
    if request.method == 'POST':
        try:
            if 'frame' not in request.FILES:
                raise ValueError("No frame data received")
            
            frame_file = request.FILES['frame']
            temp_path = os.path.join(tempfile.gettempdir(), f"frame_{int(time.time())}.jpg")
            
            with open(temp_path, 'wb+') as destination:
                for chunk in frame_file.chunks():
                    destination.write(chunk)
            
            frame = cv2.imread(temp_path)
            if frame is None:
                raise ValueError("Could not read frame data")
            
            processed_frame = preprocess_frame(frame, source='webcam')
            if processed_frame is None:
                raise ValueError("Frame preprocessing failed")
            
            input_data = np.array([processed_frame] * WEB_CAM_SETTINGS['sequence_length'], dtype=np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            
            expected_shape = tuple(tflite_input_details[0]['shape'])
            if input_data.shape != expected_shape:
                raise ValueError(f"Input shape mismatch. Expected {expected_shape}, got {input_data.shape}")
            
            tflite_interpreter.set_tensor(tflite_input_details[0]['index'], input_data)
            tflite_interpreter.invoke()
            predictions = tflite_interpreter.get_tensor(tflite_output_details[0]['index'])
            
            all_predictions = []
            for cls, conf in zip(CLASSES_LIST, predictions[0]):
                confidence = min(max(float(conf * 100), 0), 100)
                all_predictions.append({
                    'class': cls,
                    'confidence': confidence
                })
            
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            
            return JsonResponse({
                'success': True,
                'predicted_class': all_predictions[0]['class'],
                'confidence': all_predictions[0]['confidence'],
                'all_predictions': all_predictions,
                'processing_time': 2.5
            })
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
            return JsonResponse({
                'success': False,
                'error': str(e),
                'message': 'Frame processing failed'
            }, status=400)
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    }, status=405)