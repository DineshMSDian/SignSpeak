"""
SignSpeak — FastAPI Backend Server
──────────────────────────────────
WebSocket server that receives camera frames from the Flutter app,
runs MediaPipe hand detection + LSTM prediction, and returns results.

Launch:  python server.py
"""

import os
import sys
import json
import time
import socket
import asyncio
import cv2
import numpy as np
from io import BytesIO

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import config
from modules.capture_engine import HandCapture
from modules.normalization import normalize_landmarks, SequenceBuffer
from modules.predictor import ContinuousPredictor, SmoothedPredictor
from modules.model import load_trained_model

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ═══════════════════════════════════════════════════════════════
# App Setup
# ═══════════════════════════════════════════════════════════════
app = FastAPI(title="SignSpeak Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════
# Global State
# ═══════════════════════════════════════════════════════════════
import mediapipe as mp

class SignSpeakBackend:
    """Manages models, MediaPipe hands, and per-connection state."""

    def __init__(self):
        self.current_mode = "ASL"
        self.model = None
        self.label_map = None
        self.predictor = None

        # MediaPipe hands (reused across frames — not per-connection)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,      # was True match training exactly
            max_num_hands=config.NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

        self._load_model(self.current_mode)

    def _load_model(self, mode: str):
        """Load model + label map for the given mode."""
        if mode == "ASL":
            model_path = config.ASL_MODEL_PATH
            label_path = config.ASL_LABEL_MAP_PATH
        else:
            model_path = config.ISL_MODEL_PATH
            label_path = config.ISL_LABEL_MAP_PATH

        self.model = load_trained_model(model_path)
        self.label_map = None
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                self.label_map = json.load(f)

        if self.model and self.label_map:
            self.predictor = ContinuousPredictor(
                model=self.model,
                label_map=self.label_map,
                confidence_threshold=config.CONFIDENCE_THRESHOLD,
            )
            self.smoother = SmoothedPredictor(self.predictor, vote_window=5)
            print(f"[SERVER] {mode} model loaded — {len(self.label_map)} classes")
        else:
            self.predictor = None
            print(f"[SERVER] WARNING: {mode} model not found!")

        self.current_mode = mode

    def set_mode(self, mode: str):
        """Switch between ASL and ISL."""
        if mode not in ("ASL", "ISL"):
            return False
        if mode != self.current_mode:
            self._load_model(mode)
        return True

    def process_frame(self, jpeg_bytes: bytes, sequence_buffer: SequenceBuffer):
        """
        Process a single JPEG frame.

        Returns dict with prediction results.
        """
        # Decode JPEG → numpy array
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "Failed to decode frame"}

        # Flip frame horizontally to match original webcam training data orientation
        # (Otherwise left/right hand x-coordinates are reversed, breaking gestures)
        # frame = cv2.flip(frame, 1)

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.hands.process(rgb)

        # Extract landmarks
        all_landmarks = np.zeros(
            (config.NUM_HANDS * config.NUM_LANDMARKS, config.NUM_COORDS),
            dtype=np.float32,
        )
        num_hands = 0
        flutter_landmarks = []

        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= config.NUM_HANDS:
                    break
                
                hand_points = []
                for lm in hand_landmarks.landmark:
                    # MediaPipe x/y are normalized between 0.0 and 1.0
                    hand_points.append({"x": lm.x, "y": lm.y})
                flutter_landmarks.append(hand_points)

                for lm_idx, lm in enumerate(hand_landmarks.landmark):
                    offset = hand_idx * config.NUM_LANDMARKS
                    all_landmarks[offset + lm_idx] = [lm.x, lm.y, lm.z]

        # Normalize landmarks
        normalized = normalize_landmarks(all_landmarks)
        
        # The Flutter client sends every 3rd frame (~10 FPS) to save bandwidth/CPU.
        # The model was trained at 30 FPS. Duplicating the frame 3x restores the temporal ratio.
        
        for _ in range(3):
            sequence_buffer.push(normalized)

        # Base result
        result = {
            "num_hands": num_hands,
            "buffer_fill": sequence_buffer.current_length,
            "buffer_ready": sequence_buffer.is_ready(),
            "prediction": None,
            "confidence": 0.0,
            "raw_label": None,
            "landmarks": flutter_landmarks,
        }

        # Run prediction if buffer is ready
        if sequence_buffer.is_ready() and self.predictor:
            seq = sequence_buffer.get_sequence()
            label, conf = self.smoother.update(seq)
            raw_label = label
            result["prediction"] = label
            result["confidence"] = round(float(conf), 4)
            result["raw_label"] = raw_label

        return result


# Create singleton backend
backend = SignSpeakBackend()

# ═══════════════════════════════════════════════════════════════
# REST Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": backend.current_mode,
        "model_loaded": backend.model is not None,
        "classes": len(backend.label_map) if backend.label_map else 0,
    }


@app.get("/models")
def models():
    return {
        "current_mode": backend.current_mode,
        "available": list(config.SUPPORTED_LANGUAGES.keys()),
        "asl_model_exists": os.path.exists(config.ASL_MODEL_PATH),
        "isl_model_exists": os.path.exists(config.ISL_MODEL_PATH),
    }


@app.post("/set-mode/{mode}")
def set_mode(mode: str):
    mode = mode.upper()
    success = backend.set_mode(mode)
    return {
        "success": success,
        "mode": backend.current_mode,
        "classes": len(backend.label_map) if backend.label_map else 0,
        "class_names": sorted(backend.label_map.keys()) if backend.label_map else [],
    }


# ═══════════════════════════════════════════════════════════════
# WebSocket — Main Frame Processing
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[SERVER] Client connected")

    # Each connection gets its own sequence buffer
    sequence_buffer = SequenceBuffer()
    frame_count = 0

    try:
        while True:
            # Receive binary JPEG frame
            data = await websocket.receive_bytes()
            frame_count += 1

            # Process every frame (Flutter controls the send rate)
            start_time = time.perf_counter()
            result = backend.process_frame(data, sequence_buffer)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            result["frame_count"] = frame_count
            result["processing_ms"] = round(elapsed_ms, 1)
            result["mode"] = backend.current_mode

            # Send JSON result back
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("[SERVER] Client disconnected")
    except Exception as e:
        print(f"[SERVER] Error: {e}")


# ═══════════════════════════════════════════════════════════════
# WebSocket — Mode switching from client
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws/control")
async def control_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "set_mode":
                mode = data.get("mode", "").upper()
                success = backend.set_mode(mode)
                await websocket.send_json({
                    "action": "mode_changed",
                    "success": success,
                    "mode": backend.current_mode,
                    "classes": len(backend.label_map) if backend.label_map else 0,
                    "class_names": sorted(backend.label_map.keys()) if backend.label_map else [],
                })

            elif action == "reset_buffer":
                await websocket.send_json({"action": "buffer_reset", "success": True})

    except WebSocketDisconnect:
        print("[SERVER] Control client disconnected")


# ═══════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════

def get_local_ip():
    """Get the local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 8000

    print("=" * 60)
    print("  SignSpeak — Backend Server")
    print("=" * 60)
    print(f"  Mode:       {backend.current_mode}")
    print(f"  Classes:    {len(backend.label_map) if backend.label_map else 0}")
    print(f"  Local URL:  http://127.0.0.1:{port}")
    print(f"  Network:    http://{local_ip}:{port}")
    print(f"  WebSocket:  ws://{local_ip}:{port}/ws")
    print("=" * 60)
    print(f"\n  → Enter this IP in the Flutter app: {local_ip}\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
