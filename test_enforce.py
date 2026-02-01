import cv2
import numpy as np
from deepface import DeepFace
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create dummy images with no faces
dummy1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Black image
dummy2 = np.ones((100, 100, 3), dtype=np.uint8) * 200  # White image

print("Testing DeepFace with enforce_detection=True on dummy images...")
print(f"Image 1 shape: {dummy1.shape}")
print(f"Image 2 shape: {dummy2.shape}")

try:
    result = DeepFace.verify(
        dummy1, 
        dummy2,
        model_name="Facenet",
        enforce_detection=True,
        detector_backend="opencv"
    )
    print(f"✓ Result: {result}")
    print(f"  - Matched: {result.get('verified')}")
    print(f"  - Distance: {result.get('distance')}")
except Exception as e:
    print(f"✗ Exception caught: {type(e).__name__}")
    print(f"  Message: {str(e)}")
