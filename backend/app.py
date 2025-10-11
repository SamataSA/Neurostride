import os, io, base64, threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import torch
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

if os.path.exists(MODEL_PATH):
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"[INFO] Model loaded from {MODEL_PATH}")
else:
    print("[WARN] No trained model found, please run train_gait.py first")

net.eval()

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

    # Body
    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark
        body = [c for l in lm for c in (l.x, l.y)]
    else:
        body = [0.0] * BODY_DIM

    # Face + Eyes
    if results_face.multi_face_landmarks:
        lm = results_face.multi_face_landmarks[0].landmark
        # Face vector (468 landmarks → 936 values)
        face_vec = []
        for i in range(468):
            if i < len(lm):
                face_vec.extend([lm[i].x, lm[i].y])
            else:
                face_vec.extend([0.0, 0.0])
        # Eyes (subset of landmarks)
        eye_idx = [33, 133, 159, 145, 468, 473, 474, 475, 476, 477]
        eye_vec = []
        for i in eye_idx:
            if i < len(lm):
                eye_vec.extend([lm[i].x, lm[i].y])
            else:
                eye_vec.extend([0.0, 0.0])
    else:
        face_vec = [0.0] * FACE_DIM
        eye_vec = [0.0] * EYE_DIM

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
            xb = torch.tensor(np.array(buf["body"])[-SEQ_LEN:], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            xf = torch.tensor(np.array(buf["face"])[-SEQ_LEN:], dtype=torch.float32, device=DEVICE).mean(dim=0, keepdim=True)
            xe = torch.tensor(np.array(buf["eye"])[-SEQ_LEN:], dtype=torch.float32, device=DEVICE).mean(dim=0, keepdim=True)

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
