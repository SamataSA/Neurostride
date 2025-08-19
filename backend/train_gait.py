import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mediapipe as mp
import pandas as pd

# =====================
# Config
# =====================
DATASET_DIR = "dataset_gait"
MODEL_PATH = os.path.join("backend", "models", "gait_model.pth")
SEQ_LEN = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 8

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
            if i < len(lm):
                face_vec.extend([lm[i].x, lm[i].y])
            else:
                face_vec.extend([0.0, 0.0])

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

    return body, face_vec, eye_vec


def extract_features(path):
    ext = os.path.splitext(path)[-1].lower()
    body_seq, face_seq, eye_seq = [], [], []

    # Video
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

    # Image
    elif ext in [".jpg", ".jpeg", ".png"]:
        frame = cv2.imread(path)
        b, f, e = extract_from_frame(frame)
        body_seq.append(b)
        face_seq.append(f)
        eye_seq.append(e)

    # CSV
    elif ext == ".csv":
        df = pd.read_csv(path)
        # Expect CSV to have BODY_DIM + FACE_DIM + EYE_DIM columns
        for _, row in df.iterrows():
            row = row.values.astype(np.float32)
            b = row[:BODY_DIM].tolist()
            f = row[BODY_DIM:BODY_DIM+FACE_DIM].tolist()
            e = row[BODY_DIM+FACE_DIM:].tolist()
            body_seq.append(b)
            face_seq.append(f)
            eye_seq.append(e)

    # Pad / Trim
    return (
        pad_or_trim(body_seq, BODY_DIM),
        pad_or_trim(face_seq, FACE_DIM),
        pad_or_trim(eye_seq, EYE_DIM),
    )

# =====================
# Model
# =====================
class GaitNet(nn.Module):
    def __init__(self, body_dim, face_dim, eye_dim, hidden_dim=128):
        super().__init__()
        self.body_lstm = nn.LSTM(body_dim, hidden_dim, batch_first=True)
        self.face_fc = nn.Linear(face_dim, hidden_dim)
        self.eye_fc = nn.Linear(eye_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 3, 1)

    def forward(self, body_seq, face_vec, eye_vec):
        _, (h_body, _) = self.body_lstm(body_seq)
        h_face = torch.relu(self.face_fc(face_vec))
        h_eye = torch.relu(self.eye_fc(eye_vec))
        h = torch.cat([h_body[-1], h_face, h_eye], dim=-1)
        logits = self.fc_out(h)
        return logits

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
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if not os.path.isfile(path):
                continue
            if path.lower().endswith((".mp4", ".avi", ".mov", ".jpg", ".jpeg", ".png", ".csv")):
                body, facef, eye = extract_features(path)
                X_body.append(body)
                X_face.append(facef)
                X_eye.append(eye)
                y.append(0 if "normal" in label.lower() else 1)

    X_body, X_face, X_eye = map(lambda arr: np.array(arr, dtype=np.float32), (X_body, X_face, X_eye))
    y = np.array(y, dtype=np.int64)

    print(f"[INFO] Loaded {len(y)} samples.")

    # Split
    Xb_train, Xb_test, Xf_train, Xf_test, Xe_train, Xe_test, y_train, y_test = train_test_split(
        X_body, X_face, X_eye, y, test_size=0.2, stratify=y, random_state=42
    )

    net = GaitNet(BODY_DIM, FACE_DIM, EYE_DIM).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

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
            logits = net(xb, xf, xe).squeeze()
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
                logit = net(xb, xf, xe).squeeze().cpu().item()
                pred = 1 if logit > 0 else 0
                preds.append(pred)
                gts.append(y_test[j])
        acc = accuracy_score(gts, preds)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(indices):.4f} - Test Acc: {acc:.4f}")

    torch.save(net.state_dict(), MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
