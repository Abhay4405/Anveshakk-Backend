#!/usr/bin/env python
import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import cv2
import numpy as np
from urllib.parse import unquote

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

_deepface = None

def get_deepface():
    global _deepface
    if _deepface is None:
        try:
            import tensorflow as tf
            tf.get_logger().setLevel(logging.ERROR)
            from deepface import DeepFace
            _deepface = DeepFace
            logger.info("✓ DeepFace loaded")
        except Exception as e:
            logger.error(f"Failed to load DeepFace: {e}")
            raise
    return _deepface

def load_image(url: str):
    """Download image from URL"""
    try:
        url = unquote(url)
        logger.info(f"📥 Loading: {url}")
        
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Image decode failed")
        
        height, width = img.shape[:2]
        if width > 1000 or height > 1000:
            scale = 1000 / max(width, height)
            img = cv2.resize(img, (int(width * scale), int(height * scale)))
        
        logger.info(f"✓ Loaded: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        raise ValueError(f"Cannot load image: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "ok", "service": "face-matching"})

@app.route('/match-face', methods=['POST'])
def match_face():
    """Compare two faces"""
    try:
        data = request.get_json()
        
        if not data or 'img1_url' not in data or 'img2_url' not in data:
            return jsonify({"error": "Missing img1_url or img2_url"}), 400
        
        logger.info("=" * 50)
        logger.info("🔍 Face matching started")
        logger.info(f"Lost: {data['img1_url']}")
        logger.info(f"Found: {data['img2_url']}")
        
        # Load images
        img1 = load_image(data['img1_url'])
        img2 = load_image(data['img2_url'])
        
        logger.info("📊 Running verification...")
        
        # Get DeepFace
        DeepFace = get_deepface()
        
        # Verify
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="opencv",
            distance_metric="cosine"
        )
        
        distance = result.get("distance", 0)
        # Custom threshold for better matching
        # Using 0.60 threshold (more lenient for real-world images)
        verified = distance < 0.60
        confidence = max(0, (1 - distance) * 100)
        
        logger.info(f"🔍 Distance: {distance:.4f} | Threshold: 0.60 | Match: {verified}")
        
        logger.info(f"✓ Distance: {distance:.4f}")
        logger.info(f"✓ Confidence: {confidence:.2f}%")
        logger.info(f"✓ Match: {verified}")
        logger.info("=" * 50)
        
        return jsonify({
            "matched": verified,
            "confidence": round(confidence, 2),
            "distance": round(distance, 4)
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def root():
    return jsonify({"service": "Face Matching API", "status": "running"})

if __name__ == '__main__':
    logger.info("🚀 Starting Face Matching API on http://0.0.0.0:9000")
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=True)
