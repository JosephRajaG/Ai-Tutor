"""
Camera utilities for webcam frame capture.
"""
import cv2
import numpy as np
from typing import Optional, Tuple


class Camera:
    """Handles webcam capture and frame processing."""
    
    def __init__(self, camera_index: int = 0):
        """
        Initialize camera.
        
        Args:
            camera_index: Index of the camera device (default: 0)
        """
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_active = False
    
    def start(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                self.is_active = True
                return True
            else:
                return False
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture and release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_active = False
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from the camera.
        
        Returns:
            Frame as numpy array (BGR format), or None if failed
        """
        if not self.is_active or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def get_frame_rgb(self) -> Optional[np.ndarray]:
        """
        Read a frame and convert to RGB format.
        
        Returns:
            Frame as numpy array (RGB format), or None if failed
        """
        frame = self.read_frame()
        if frame is not None:
            # OpenCV uses BGR, convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        return None

