import os, io, base64, threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from model import GaitModel
from alert import send_alert  # NEW: Alert system

# =====================
# Paths
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =====================
# Flask app
# =====================
app = Flask(__name__, static_folder=FRONTEND_DIR, template_folder=FRONTEND_DIR)
CORS(app)

# =====================
# MediaPipe Pose
# =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# =====================
# Model + Buffers
# =====================
MODEL_PATH = os.path.join(MODELS_DIR, "gait_model.pth")
GALLERY_PATH = os.path.join(MODELS_DIR, "gallery.npz")
DEVICE = "cuda" if False else "cpu"

gait_model = GaitModel(model_path=MODEL_PATH, gallery_path=GALLERY_PATH, device=DEVICE)

SEQ_LEN = 48
buffers = {}
lock = threading.Lock()

# =====================
# Routes for Frontend
# =====================
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "login.html")

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(FRONTEND_DIR, "favicon.ico")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(FRONTEND_DIR, path)

# =====================
# API Endpoints
# =====================
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": gait_model.is_loaded,
        "gallery": gait_model.gallery_info(),
        "seq_len": SEQ_LEN,
    })

@app.route("/reset_buffer", methods=["POST"])
def reset_buffer():
    session_id = request.json.get("session_id", "default_session")
    with lock:
        buffers[session_id] = []
    return jsonify({"status": "cleared", "session_id": session_id})

@app.route("/analyze_frame", methods=["POST"])
def analyze_frame_route():
    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    session_id = data.get("session_id", "default_session")
    force = bool(data.get("force", False))

    if not img_b64:
        return jsonify({"error": "no image provided"}), 400

    try:
        if "," in img_b64:
            _, img_b64 = img_b64.split(",", 1)
        img_bytes = base64.b64decode(img_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(pil_img)[:, :, ::-1]
    except Exception as e:
        return jsonify({"error": f"bad image: {e}"}), 400

    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return jsonify({
            "matched": False,
            "confidence": 0.0,
            "suspect_id": None,
            "similarity": 0.0,
            "top_k": [],
            "suspicious_behaviors": [],
            "note": "no_pose"
        }), 200

    lm = results.pose_landmarks.landmark
    keypoints = []
    for l in lm:
        keypoints.extend([l.x, l.y])
    keypoints = np.asarray(keypoints, dtype=np.float32)

    with lock:
        buf = buffers.setdefault(session_id, [])
        buf.append(keypoints)
        if len(buf) > SEQ_LEN:
            buf.pop(0)
        have = len(buf)

    if have >= SEQ_LEN or force:
        with lock:
            seq = np.stack(buf[-SEQ_LEN:], axis=0) if have >= SEQ_LEN else np.stack(buf, axis=0)
        result = gait_model.infer_and_match(seq)
        result.update({"have": have, "need": SEQ_LEN})

        # =====================
        # ALERT SYSTEM TRIGGER
        # =====================
        if result["matched"] or len(result["suspicious_behaviors"]) > 0:
            alert_image_path = os.path.join(BASE_DIR, "last_detected.jpg")
            cv2.imwrite(alert_image_path, img)
            send_alert(
                suspect_id=result["suspect_id"],
                similarity=result["similarity"],
                behaviors=result["suspicious_behaviors"],
                image_path=alert_image_path
            )

        return jsonify(result)
    else:
        return jsonify({"status": "buffering", "have": have, "need": SEQ_LEN})

# =====================
# Run server
# =====================
if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
