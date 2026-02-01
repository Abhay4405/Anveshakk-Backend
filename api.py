from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import cv2  
import numpy as np
import logging
from urllib.parse import unquote
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="Face Matching API")

# Add CORS
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

# Global DeepFace instance
_deepface_instance = None

def get_deepface_module():
    """Lazy load DeepFace module"""
    global _deepface_instance
    if _deepface_instance is None:
        try:
            import tensorflow as tf
            tf.get_logger().setLevel(logging.ERROR)
            from deepface import DeepFace
            _deepface_instance = DeepFace
            logger.info("✓ DeepFace loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DeepFace: {e}")
            raise
    return _deepface_instance

def load_image(url: str):
    """Download and decode image from URL"""
    try:
        url = unquote(url)
        logger.info(f"📥 Loading image from: {url}")
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Image decode failed")
        
        # Resize if too large
        height, width = img.shape[:2]
        if width > 1000 or height > 1000:
            scale = 1000 / max(width, height)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        logger.info(f"✓ Image loaded: {img.shape}")
        return img
        
    except Exception as e:
        logger.error(f"❌ Image loading failed: {str(e)}")
        raise ValueError(f"Cannot load image: {str(e)}")

@app.get("/")
def root():
    """Root endpoint"""
    return {"service": "Face Matching API", "status": "running"}

@app.get("/health")
def health():
    """Health check"""
    return {"status": "ok", "service": "face-matching"}

@app.post("/match-face")
def match_face(request: MatchRequest):
    """Compare two face images"""
    try:
        logger.info("=" * 60)
        logger.info("🔍 FACE MATCHING REQUEST")
        logger.info(f"Lost image: {request.img1_url}")
        logger.info(f"Found image: {request.img2_url}")
        
        # Load images
        logger.info("📥 Loading images...")
        img1 = load_image(request.img1_url)
        img2 = load_image(request.img2_url)
        logger.info(f"✓ Image 1 loaded: {img1.shape}")
        logger.info(f"✓ Image 2 loaded: {img2.shape}")
        
        logger.info("🔎 Running DeepFace.verify() with ENFORCE_DETECTION=True...")
        
        # Get DeepFace
        DeepFace = get_deepface_module()
        
        # Verify faces (ENFORCE DETECTION - don't match if no face found)
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name="Facenet",
            enforce_detection=True,
            detector_backend="opencv",
            align=True,
            normalization="base"
        )
        
        distance = result.get("distance", float('inf'))
        verified = result.get("verified", False)
        confidence = (1 - distance) * 100 if distance != float('inf') else 0
        
        # Threshold: 0.6 distance = 40% confidence (Facenet standard threshold)
        # Only match if distance <= 0.6 (confidence >= 40%)
        is_match = verified and distance <= 0.6
        
        logger.info(f"\n✓ Face Detection: SUCCESS")
        logger.info(f"✓ Distance: {distance:.4f}")
        logger.info(f"✓ Confidence: {confidence:.2f}%")
        logger.info(f"✓ Threshold check (distance <= 0.6): {is_match}")
        logger.info(f"✓ FINAL RESULT: {'MATCHED' if is_match else 'NO MATCH'}")
        logger.info("=" * 60)
        
        return {
            "matched": is_match,
            "confidence": round(confidence, 2),
            "distance": round(distance, 4)
        }
        
    except Exception as e:
        logger.error(f"❌ FACE MATCHING FAILED!")
        logger.error(f"❌ Error Type: {type(e).__name__}")
        logger.error(f"❌ Error Message: {str(e)}")
        logger.error("=" * 60)
        raise HTTPException(status_code=400, detail=f"Face matching error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting Face Matching API on http://0.0.0.0:9000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info",
        access_log=True
    )
