import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import mediapipe as mp

from model import GaitNet, _normalize_seq, INPUT_DIM, EMB_DIM

# =====================
# Paths
# =====================
DATASET_DIR = "dataset_gait"
MODEL_PATH = os.path.join("backend", "models", "gait_model.pth")
GALLERY_PATH = os.path.join("backend", "models", "gallery.npz")
SEQ_LEN = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# =====================
# MediaPipe Pose
# =====================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# =====================
# Helpers
# =====================
def extract_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            keypoints = []
            for l in lm:
                keypoints.extend([l.x, l.y])
            seq.append(keypoints)
    cap.release()

    seq = np.array(seq, dtype=np.float32)
    if len(seq) >= SEQ_LEN:
        seq = seq[:SEQ_LEN]
    elif len(seq) > 0:
        pad = np.zeros((SEQ_LEN - len(seq), INPUT_DIM), dtype=np.float32)
        seq = np.vstack([seq, pad])
    else:
        return None
    return seq

# =====================
# Load dataset
# =====================
print("[INFO] Loading dataset...")
X, y = [], []
person_ids = sorted(os.listdir(DATASET_DIR))
for pid in person_ids:
    folder = os.path.join(DATASET_DIR, pid)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.lower().endswith((".mp4", ".avi", ".mov")):
            path = os.path.join(folder, file)
            seq = extract_sequence(path)
            if seq is not None:
                X.append(seq)
                y.append(pid)

X = np.array(X, dtype=np.float32)
y = np.array(y)
id_to_idx = {pid: idx for idx, pid in enumerate(sorted(set(y)))}
y_idx = np.array([id_to_idx[label] for label in y], dtype=np.int64)

print(f"[INFO] Loaded {len(X)} sequences from {len(set(y))} people.")

if len(X) == 0:
    raise RuntimeError("No valid data found.")

# Normalize sequences like backend
X_norm = np.array([_normalize_seq(seq) for seq in X], dtype=np.float32)

# =====================
# Train/Test Split
# =====================
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_idx, test_size=0.2, random_state=42)

# =====================
# Model + Training
# =====================
net = GaitNet(input_dim=INPUT_DIM, emb_dim=EMB_DIM, num_classes=len(set(y_idx))).to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("[INFO] Training model...")
for epoch in range(20):
    net.train()
    total_loss = 0
    for i in range(len(X_train)):
        xb = torch.tensor(X_train[i:i+1], dtype=torch.float32, device=DEVICE)
        yb = torch.tensor([y_train[i]], dtype=torch.long, device=DEVICE)
        optimizer.zero_grad()
        _, logits = net(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(X_train):.4f}")

# =====================
# Save model
# =====================
torch.save(net.state_dict(), MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")

# =====================
# Create gallery embeddings
# =====================
print("[INFO] Creating gallery embeddings...")
net.eval()
embeddings = []
ids = []
with torch.no_grad():
    for seq, label in zip(X_norm, y):
        xb = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32, device=DEVICE)
        emb = net.encoder(xb).cpu().numpy()[0]
        embeddings.append(emb)
        ids.append(label)

embeddings = np.array(embeddings, dtype=np.float32)
np.savez(GALLERY_PATH, ids=np.array(ids), embeddings=embeddings)
print(f"[INFO] Gallery saved to {GALLERY_PATH}")
