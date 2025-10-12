import os, io, base64, threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from train_gait import GaitNet, BODY_DIM, FACE_DIM, EYE_DIM, SEQ_LEN

# =====================
# Paths
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "gait_model.pth")

# =====================
# Flask app
# =====================
app = Flask(__name__, static_folder=FRONTEND_DIR, template_folder=FRONTEND_DIR)
CORS(app)

# =====================
# MediaPipe Setup
# =====================
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face = mp_face.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

# =====================
# Load Model
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
net = GaitNet(BODY_DIM, FACE_DIM, EYE_DIM).to(DEVICE)

def safe_load_model(model, checkpoint_path):
    """Load checkpoint safely with proper initialization for mismatched layers."""
    if not os.path.exists(checkpoint_path):
        print("[WARN] No trained model found, please run train_gait.py first")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model_dict = model.state_dict()

    for k, v in checkpoint.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                model_dict[k] = v
            else:
                # Initialize weights properly
                if model_dict[k].dim() >= 2:
                    nn.init.xavier_uniform_(model_dict[k])
                else:
                    nn.init.zeros_(model_dict[k])
                print(f"[WARN] Layer {k} shape mismatch, reinitialized")
        else:
            print(f"[WARN] Checkpoint key {k} not in model, skipping")

    # Initialize missing layers
    for k in model_dict.keys():
        if k not in checkpoint:
            if model_dict[k].dim() >= 2:
                nn.init.xavier_uniform_(model_dict[k])
            else:
                nn.init.zeros_(model_dict[k])
            print(f"[INFO] Initialized missing layer {k}")

    model.load_state_dict(model_dict)
    model.eval()
    print(f"[INFO] Model loaded from {checkpoint_path}")

safe_load_model(net, MODEL_PATH)

# =====================
# Thread-safe buffer
# =====================
buffers = {}
lock = threading.Lock()

def normalize_seq(seq):
    """Normalize each feature to [0,1] based on min/max across the sequence."""
    arr = np.array(seq, dtype=np.float32)
    min_val = arr.min(axis=0, keepdims=True)
    max_val = arr.max(axis=0, keepdims=True)
    denom = np.maximum(max_val - min_val, 1e-6)
    return (arr - min_val) / denom

def moving_average(seq, window=3):
    """Simple moving average smoothing."""
    arr = np.array(seq, dtype=np.float32)
    if len(arr) < window:
        return arr
    smoothed = np.convolve(arr.flatten(), np.ones(window)/window, mode='same')
    return smoothed.reshape(arr.shape)

# =====================
# Frontend routes
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
        "model_loaded": os.path.exists(MODEL_PATH),
        "seq_len": SEQ_LEN,
    })

@app.route("/reset_buffer", methods=["POST"])
def reset_buffer():
    session_id = request.json.get("session_id", "default_session")
    with lock:
        buffers[session_id] = {"body": [], "face": [], "eye": []}
    return jsonify({"status": "cleared", "session_id": session_id})

@app.route("/analyze_frame", methods=["POST"])
def analyze_frame_route():
    data = request.get_json(force=True)
    img_b64 = data.get("image", "")
    session_id = data.get("session_id", "default_session")
    force = bool(data.get("force", False))

    if not img_b64:
        return jsonify({"error": "no image provided"}), 400

    # Decode base64 → image
    try:
        if "," in img_b64:
            _, img_b64 = img_b64.split(",", 1)
        img_bytes = base64.b64decode(img_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = np.array(pil_img)[:, :, ::-1]  # RGB → BGR
    except Exception as e:
        return jsonify({"error": f"bad image: {e}"}), 400

    # Extract landmarks
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(frame_rgb)
    results_face = face.process(frame_rgb)

    # Body landmarks
    body = [0.0] * BODY_DIM
    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark
        body = [c for l in lm for c in (l.x, l.y)]

    # Face + Eyes
    face_vec = [0.0] * FACE_DIM
    eye_vec = [0.0] * EYE_DIM
    if results_face.multi_face_landmarks:
        lm = results_face.multi_face_landmarks[0].landmark
        face_vec = []
        for i in range(468):
            face_vec.extend([lm[i].x, lm[i].y] if i < len(lm) else [0.0, 0.0])
        eye_idx = [33, 133, 159, 145, 468, 473, 474, 475, 476, 477]
        eye_vec = []
        for i in eye_idx:
            eye_vec.extend([lm[i].x, lm[i].y] if i < len(lm) else [0.0, 0.0])

    # =====================
    # Buffering sequence
    # =====================
    with lock:
        buf = buffers.setdefault(session_id, {"body": [], "face": [], "eye": []})
        buf["body"].append(body)
        buf["face"].append(face_vec)
        buf["eye"].append(eye_vec)
        if len(buf["body"]) > SEQ_LEN:
            buf["body"].pop(0)
            buf["face"].pop(0)
            buf["eye"].pop(0)
        have = len(buf["body"])

    # =====================
    # Run model if enough frames
    # =====================
    if have >= SEQ_LEN or force:
        with lock:
            xb = normalize_seq(buf["body"][-SEQ_LEN:])
            xb = torch.tensor(xb, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            xf = normalize_seq(buf["face"][-SEQ_LEN:])
            xf = moving_average(xf)
            xf = torch.tensor(xf.mean(axis=0, keepdim=True), dtype=torch.float32, device=DEVICE)

            xe = normalize_seq(buf["eye"][-SEQ_LEN:])
            xe = moving_average(xe)
            xe = torch.tensor(xe.mean(axis=0, keepdim=True), dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            logit = net(xb, xf, xe).squeeze().cpu().item()
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            pred = 1 if logit > 0 else 0

        return jsonify({
            "status": "done",
            "have": have,
            "need": SEQ_LEN,
            "prediction": "thief" if pred == 1 else "normal",
            "confidence": prob
        })
    else:
        return jsonify({"status": "buffering", "have": have, "need": SEQ_LEN})

# =====================
# Run server
# =====================
if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
