"""
Emotion detection using ONNX model.
"""
import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional, Dict
import os


class EmotionDetector:
    """Detects emotions from facial images using ONNX model."""
    
    # Emotion labels (standard order for most emotion models)
    EMOTION_LABELS = ['angry', 'confused', 'frustrated', 'happy', 'neutral', 'sad']
    
    def __init__(self, model_path: str = "models/emotion.onnx"):
        """
        Initialize emotion detector.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.face_cascade = None
        self.input_size = (48, 48)  # Default size, will be inferred from model
        self.expected_channels = 1  # Default to grayscale, will be inferred from model
        self.input_shape = None
        self._load_model()
        self._load_face_detector()
    
    def _load_model(self):
        """Load ONNX model and infer input/output shapes."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please place your emotion.onnx file in the models/ directory."
            )
        
        try:
            # Create inference session
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get input shape from model
            input_details = self.session.get_inputs()[0]
            input_shape = input_details.shape
            
            # Determine if model expects RGB (3 channels) or grayscale (1 channel)
            if len(input_shape) == 4:  # [batch, channels, height, width] or [batch, height, width, channels]
                # Check channel position - typically channels are at index 1 or 3
                # If index 1 is small (1-4), it's likely [batch, channels, height, width]
                # If index 1 is large (like 260), it's likely [batch, height, width, channels]
                if input_shape[1] is not None and input_shape[1] <= 4:
                    # Format: [batch, channels, height, width]
                    self.expected_channels = int(input_shape[1])
                    if input_shape[2] is not None and input_shape[3] is not None:
                        self.input_size = (int(input_shape[3]), int(input_shape[2]))
                    else:
                        self.input_size = (260, 260)  # Default fallback
                elif input_shape[3] is not None and input_shape[3] <= 4:
                    # Format: [batch, height, width, channels]
                    self.expected_channels = int(input_shape[3])
                    if input_shape[1] is not None and input_shape[2] is not None:
                        self.input_size = (int(input_shape[2]), int(input_shape[1]))
                    else:
                        self.input_size = (260, 260)  # Default fallback
                else:
                    # Can't determine, default to RGB (3 channels)
                    self.expected_channels = 3
                    if input_shape[2] is not None and input_shape[3] is not None:
                        self.input_size = (int(input_shape[3]), int(input_shape[2]))
                    else:
                        self.input_size = (260, 260)
            else:
                self.expected_channels = 1  # Assume grayscale for non-4D inputs
                if len(input_shape) >= 2:
                    if input_shape[-2] is not None and input_shape[-1] is not None:
                        self.input_size = (input_shape[-1], input_shape[-2])
                    else:
                        self.input_size = (260, 260)
                else:
                    self.input_size = (260, 260)
            
            # Store input shape for later use
            self.input_shape = input_shape
            print(f"Model loaded successfully. Input size: {self.input_size}, Expected channels: {self.expected_channels}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def _load_face_detector(self):
        """Load OpenCV Haar cascade for face detection."""
        try:
            # Try to load the default face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                print("Warning: Could not load face cascade. Will use full frame.")
                self.face_cascade = None
        except Exception as e:
            print(f"Warning: Could not load face cascade: {e}")
            self.face_cascade = None
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame.
        
        Args:
            frame: Input frame (BGR or RGB)
            
        Returns:
            Tuple of (x, y, width, height) if face found, None otherwise
        """
        if self.face_cascade is None:
            # Return full frame if no cascade available
            h, w = frame.shape[:2]
            return (0, 0, w, h)
        
        # Convert to grayscale for face detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.shape[2] == 3 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Return the largest face
            largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
            return tuple(largest_face)
        
        return None
    
    def preprocess_frame(self, frame: np.ndarray, face_rect: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Preprocess frame for emotion detection.
        
        Args:
            frame: Input frame (RGB or BGR)
            face_rect: Face bounding box (x, y, width, height), or None to use full frame
            
        Returns:
            Preprocessed image ready for model input, or None if failed
        """
        try:
            # Extract face region if provided
            if face_rect is not None:
                x, y, w, h = face_rect
                face_roi = frame[y:y+h, x:x+w]
            else:
                face_roi = frame
            
            # Ensure frame is in RGB format (model expects RGB)
            if len(face_roi.shape) == 2:
                # If grayscale, convert to RGB by duplicating channels
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)
            elif len(face_roi.shape) == 3:
                # Already has channels, ensure it's RGB (not BGR)
                if face_roi.shape[2] == 3:
                    # Check if values are in 0-255 range (BGR from OpenCV) or 0-1 range
                    if np.max(face_roi) > 1.0:
                        # Likely BGR from OpenCV, convert to RGB
                        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    # If already in 0-1 range, assume it's already RGB
            
            # Resize to model input size
            resized = cv2.resize(face_roi, self.input_size, interpolation=cv2.INTER_AREA)
            
            # Normalize to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            # Reshape for model input based on expected format
            input_details = self.session.get_inputs()[0]
            input_shape = input_details.shape
            
            if len(input_shape) == 4:
                # Check if format is [batch, channels, height, width] or [batch, height, width, channels]
                if input_shape[1] is not None and input_shape[1] <= 4:
                    # Format: [batch, channels, height, width]
                    if self.expected_channels == 3:
                        # RGB: (1, 3, H, W)
                        normalized = normalized.transpose(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                        normalized = normalized.reshape(1, 3, self.input_size[1], self.input_size[0])
                    else:
                        # Grayscale: (1, 1, H, W)
                        gray = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        normalized = (gray.astype(np.float32) / 255.0).reshape(1, 1, self.input_size[1], self.input_size[0])
                else:
                    # Format: [batch, height, width, channels]
                    if self.expected_channels == 3:
                        # RGB: (1, H, W, 3)
                        normalized = normalized.reshape(1, self.input_size[1], self.input_size[0], 3)
                    else:
                        # Grayscale: (1, H, W, 1)
                        gray = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                        normalized = (gray.astype(np.float32) / 255.0).reshape(1, self.input_size[1], self.input_size[0], 1)
            elif len(input_shape) == 3:
                # Add batch dimension: (1, H, W) or (1, H, W, C)
                if self.expected_channels == 3:
                    normalized = normalized.reshape(1, self.input_size[1], self.input_size[0], 3)
                else:
                    gray = cv2.cvtColor((normalized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                    normalized = (gray.astype(np.float32) / 255.0).reshape(1, self.input_size[1], self.input_size[0])
            elif len(input_shape) == 2:
                # Flatten: (1, H*W*C)
                normalized = normalized.reshape(1, -1)
            
            return normalized
            
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None
    
    def predict_emotion(self, frame: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict emotion from frame.
        
        Args:
            frame: Input frame (RGB format)
            
        Returns:
            Tuple of (emotion_label, confidence, all_probabilities_dict)
        """
        if self.session is None:
            return "neutral", 0.0, {}
        
        # Detect face - if no face found, use center crop of frame
        face_rect = self.detect_face(frame)
        if face_rect is None:
            # If no face detected, use center region of frame
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            crop_size = min(w, h) // 2
            face_rect = (
                max(0, center_x - crop_size // 2),
                max(0, center_y - crop_size // 2),
                crop_size,
                crop_size
            )
        
        # Preprocess
        preprocessed = self.preprocess_frame(frame, face_rect)
        if preprocessed is None:
            return "neutral", 0.0, {}
        
        try:
            # Get input name
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            # Run inference
            outputs = self.session.run([output_name], {input_name: preprocessed})
            predictions = outputs[0]
            
            # Handle different output formats
            if len(predictions.shape) > 1:
                predictions = predictions[0]  # Remove batch dimension
            
            # Get probabilities
            if predictions.dtype != np.float32:
                predictions = predictions.astype(np.float32)
            
            # Apply softmax if needed (if outputs are logits)
            if np.any(predictions < 0) or np.sum(predictions) < 0.9:
                # Likely logits, apply softmax
                exp_preds = np.exp(predictions - np.max(predictions))
                probabilities = exp_preds / np.sum(exp_preds)
            else:
                # Normalize to probabilities
                probabilities = predictions / np.sum(predictions) if np.sum(predictions) > 0 else predictions
            
            # Map to emotion labels
            # Handle case where model has different number of outputs
            num_emotions = len(self.EMOTION_LABELS)
            if len(probabilities) != num_emotions:
                # If model has different output, map to available labels
                if len(probabilities) >= num_emotions:
                    probabilities = probabilities[:num_emotions]
                else:
                    # Pad with zeros
                    padded = np.zeros(num_emotions)
                    padded[:len(probabilities)] = probabilities
                    probabilities = padded / np.sum(padded) if np.sum(padded) > 0 else padded
            
            # Create probability dictionary
            prob_dict = {
                label: float(prob) 
                for label, prob in zip(self.EMOTION_LABELS, probabilities)
            }
            
            # Get predicted emotion (only if confidence is above threshold)
            predicted_idx = np.argmax(probabilities)
            predicted_emotion = self.EMOTION_LABELS[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Only return non-neutral if confidence is reasonable
            if confidence < 0.15 and predicted_emotion == "neutral":
                # If all emotions are low confidence, check second highest
                sorted_indices = np.argsort(probabilities)[::-1]
                if len(sorted_indices) > 1:
                    second_idx = sorted_indices[1]
                    second_confidence = float(probabilities[second_idx])
                    # If second emotion is close, use it if it's more interesting
                    if second_confidence > 0.1 and abs(confidence - second_confidence) < 0.1:
                        predicted_emotion = self.EMOTION_LABELS[second_idx]
                        confidence = second_confidence
            
            return predicted_emotion, confidence, prob_dict
            
        except Exception as e:
            print(f"Error during emotion prediction: {e}")
            import traceback
            traceback.print_exc()
            return "neutral", 0.0, {}

