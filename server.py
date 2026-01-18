from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import cv2
import numpy as np
import logging
from urllib.parse import unquote
import os

# Suppress TensorFlow warnings BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Lazy imports to avoid startup issues
logging_setup = False

def get_deepface():
    """Lazy load DeepFace to avoid startup issues"""
    global logging_setup
    if not logging_setup:
        import tensorflow as tf
        tf.get_logger().setLevel(logging.ERROR)
        logging_setup = True
    from deepface import DeepFace
    return DeepFace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    img1_url: str
    img2_url: str

def load_image(url: str):
    """Download and decode image from URL"""
    try:
        url = unquote(url)
        logger.info(f"Loading image from: {url}")
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Image decode failed")
        
        # Resize image
        height, width = img.shape[:2]
        if width > 1000 or height > 1000:
            scale = 1000 / max(width, height)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        logger.info(f"Image loaded: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"Image loading failed: {str(e)}")
        raise ValueError(f"Cannot load image: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "face-matching-api"}

@app.post("/match-face")
def match_face(data: MatchRequest):
    """Compare two faces using DeepFace"""
    try:
        logger.info("Starting face matching...")
        logger.info(f"Image 1 (Lost): {data.img1_url}")
        logger.info(f"Image 2 (Found): {data.img2_url}")
        
        # Load both images
        img1 = load_image(data.img1_url)
        img2 = load_image(data.img2_url)
        
        logger.info("Images loaded successfully. Running DeepFace verification...")
        
        # Get DeepFace (lazy load)
        DeepFace = get_deepface()
        
        # Run DeepFace with proper settings
        result = DeepFace.verify(
            img1,
            img2,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv"
        )
        
        # Calculate confidence
        distance = result["distance"]
        confidence = (1 - distance) * 100
        
        # Ensure matched is boolean
        matched = bool(result["verified"])
        
        logger.info(f"✓ Distance: {distance}")
        logger.info(f"✓ Confidence: {confidence}%")
        logger.info(f"✓ Matched: {matched}")
        
        return {
            "matched": matched,
            "confidence": round(confidence, 2),
            "distance": round(distance, 4)
        }
        
    except Exception as e:
        logger.error(f"Face matching error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Face Matching API on http://0.0.0.0:9000")
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
