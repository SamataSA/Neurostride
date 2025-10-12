import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import random

# =====================
# Config
# =====================
DATASET_DIR = "dataset_gait"
MODEL_PATH = os.path.join("backend", "models", "gait_model.pth")
SEQ_LEN = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
PATIENCE = 5
THRESHOLD = 0.7  # probability threshold for "thief" prediction

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# =====================
# MediaPipe Setup
# =====================
mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face = mp_face.FaceMesh(static_image_mode=False, refine_landmarks=True, max_num_faces=1)

BODY_DIM = 33 * 2
FACE_DIM = 468 * 2
EYE_DIM = 10 * 2

# =====================
# Helpers
# =====================
def pad_or_trim(seq, dim):
    arr = np.array(seq, dtype=np.float32)
    if len(arr) >= SEQ_LEN:
        return arr[:SEQ_LEN]
    pad = np.zeros((SEQ_LEN - len(arr), dim), dtype=np.float32)
    return np.vstack([arr, pad])

def normalize(seq):
    seq = np.array(seq, dtype=np.float32)
    seq -= np.mean(seq, axis=0, keepdims=True)
    std = np.std(seq, axis=0, keepdims=True) + 1e-6
    seq /= std
    return seq

def augment_sequence(body, face, eye):
    """Random augmentation: small jitter, horizontal flip"""
    if random.random() < 0.5:
        # Horizontal flip (x → 1-x)
        body[:, 0] = 1 - body[:, 0]
        face[:, 0] = 1 - face[:, 0]
        eye[:, 0] = 1 - eye[:, 0]
    # Small Gaussian noise
    body += np.random.normal(0, 0.005, body.shape)
    face += np.random.normal(0, 0.005, face.shape)
    eye += np.random.normal(0, 0.005, eye.shape)
    return body, face, eye

# =====================
# Feature Extraction
# =====================
def extract_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Body
    results_pose = pose.process(frame_rgb)
    if results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark
        body = [c for l in lm for c in (l.x, l.y)]
    else:
        body = [0.0] * BODY_DIM

    # Face + Eye
    results_face = face.process(frame_rgb)
    if results_face.multi_face_landmarks:
        lm = results_face.multi_face_landmarks[0].landmark
        face_vec = []
        for i in range(468):
            face_vec.extend([lm[i].x, lm[i].y] if i < len(lm) else [0.0, 0.0])
        eye_idx = [33, 133, 159, 145, 468, 473, 474, 475, 476, 477]
        eye_vec = []
        for i in eye_idx:
            eye_vec.extend([lm[i].x, lm[i].y] if i < len(lm) else [0.0, 0.0])
    else:
        face_vec = [0.0] * FACE_DIM
        eye_vec = [0.0] * EYE_DIM

    return body, face_vec, eye_vec

def extract_features(path):
    ext = os.path.splitext(path)[-1].lower()
    body_seq, face_seq, eye_seq = [], [], []

    if ext in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            b, f, e = extract_from_frame(frame)
            body_seq.append(b)
            face_seq.append(f)
            eye_seq.append(e)
        cap.release()
    elif ext in [".jpg", ".jpeg", ".png"]:
        frame = cv2.imread(path)
        b, f, e = extract_from_frame(frame)
        body_seq.append(b)
        face_seq.append(f)
        eye_seq.append(e)
    elif ext == ".csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            row = row.values.astype(np.float32)
            b = row[:BODY_DIM].tolist()
            f = row[BODY_DIM:BODY_DIM+FACE_DIM].tolist()
            e = row[BODY_DIM+FACE_DIM:].tolist()
            body_seq.append(b)
            face_seq.append(f)
            eye_seq.append(e)

    # Normalize & pad
    body_seq = normalize(pad_or_trim(body_seq, BODY_DIM))
    face_seq = normalize(pad_or_trim(face_seq, FACE_DIM))
    eye_seq = normalize(pad_or_trim(eye_seq, EYE_DIM))
    return body_seq, face_seq, eye_seq

# =====================
# Model
# =====================
class GaitNet(nn.Module):
    def __init__(self, body_dim, face_dim, eye_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        # single-layer LSTM (dropout removed to avoid warning)
        self.body_lstm = nn.LSTM(body_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.0)
        self.face_fc = nn.Sequential(
            nn.Linear(face_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.eye_fc = nn.Sequential(
            nn.Linear(eye_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, body_seq, face_vec, eye_vec):
        _, (h_body, _) = self.body_lstm(body_seq)
        h_body = torch.cat((h_body[-2], h_body[-1]), dim=-1)  # BiLSTM
        h_face = self.face_fc(face_vec)
        h_eye = self.eye_fc(eye_vec)
        h = torch.cat([h_body, h_face, h_eye], dim=-1)
        logits = self.fc_out(h)
        return logits.squeeze()

# =====================
# Training
# =====================
def main():
    print("[INFO] Loading dataset...")
    X_body, X_face, X_eye, y = [], [], [], []

    for label in os.listdir(DATASET_DIR):
        folder = os.path.join(DATASET_DIR, label)
        if not os.path.isdir(folder):
            continue
        for file in tqdm(os.listdir(folder), desc=f"Loading {label}"):
            path = os.path.join(folder, file)
            if not os.path.isfile(path):
                continue
            if path.lower().endswith((".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".csv")):
                body, facef, eye = extract_features(path)
                # Apply augmentation only on training data
                if "train" in path.lower():
                    body, facef, eye = augment_sequence(body, facef, eye)
                X_body.append(body)
                X_face.append(facef)
                X_eye.append(eye)
                y.append(0 if "normal" in label.lower() else 1)

    X_body, X_face, X_eye = map(lambda arr: np.array(arr, dtype=np.float32), (X_body, X_face, X_eye))
    y = np.array(y, dtype=np.int64)
    print(f"[INFO] Loaded {len(y)} samples.")

    # Split dataset
    Xb_train, Xb_test, Xf_train, Xf_test, Xe_train, Xe_test, y_train, y_test = train_test_split(
        X_body, X_face, X_eye, y, test_size=0.25, stratify=y, random_state=42
    )

    # Compute class weights
    thief_count = np.sum(y_train==1)
    normal_count = np.sum(y_train==0)
    pos_weight = torch.tensor([normal_count / max(thief_count,1)], dtype=torch.float32, device=DEVICE)

    net = GaitNet(BODY_DIM, FACE_DIM, EYE_DIM).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    best_acc, counter = 0, 0
    print("[INFO] Training model...")

    for epoch in range(EPOCHS):
        net.train()
        indices = np.arange(len(Xb_train))
        np.random.shuffle(indices)
        total_loss = 0

        for i in range(0, len(indices), BATCH_SIZE):
            batch_idx = indices[i:i+BATCH_SIZE]
            xb = torch.tensor(Xb_train[batch_idx], device=DEVICE)
            xf = torch.tensor(Xf_train[batch_idx][:,0,:], device=DEVICE)
            xe = torch.tensor(Xe_train[batch_idx][:,0,:], device=DEVICE)
            yb = torch.tensor(y_train[batch_idx], dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            logits = net(xb, xf, xe)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        net.eval()
        preds, gts = [], []
        with torch.no_grad():
            for j in range(len(Xb_test)):
                xb = torch.tensor(Xb_test[j:j+1], device=DEVICE)
                xf = torch.tensor(Xf_test[j:j+1,0,:], device=DEVICE)
                xe = torch.tensor(Xe_test[j:j+1,0,:], device=DEVICE)
                logit = net(xb, xf, xe).cpu().item()
                prob = 1 / (1 + np.exp(-logit))
                preds.append(1 if prob > THRESHOLD else 0)
                gts.append(int(y_test[j]))

        acc = accuracy_score(gts, preds)
        scheduler.step(acc)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(indices):.4f} - Val Acc: {acc:.4f}")

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            counter = 0
            torch.save(net.state_dict(), MODEL_PATH)
            print(f"[INFO] ✅ Accuracy improved. Model saved to {MODEL_PATH}")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("[INFO] Early stopping triggered.")
                break

    print(f"\n[INFO] Training complete. Best accuracy: {best_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(gts, preds, target_names=["Normal", "Thief"]))


if __name__ == "__main__":
    main()
