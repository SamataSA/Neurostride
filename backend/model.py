# backend/model.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

# ======================
# Constants
# ======================
NUM_LM = 33           # MediaPipe pose landmarks
FEAT_PER_LM = 2       # Using only x, y
INPUT_DIM = NUM_LM * FEAT_PER_LM  # 66
EMB_DIM = 128

# ======================
# Network Definitions
# ======================
class PoseSeqEncoder(nn.Module):
    """Encodes a sequence of 2D pose keypoints into an embedding."""
    def __init__(self, input_dim=INPUT_DIM, emb_dim=EMB_DIM, hidden=192, num_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(hidden * 2, emb_dim)

    def forward(self, x):
        h = torch.relu(self.proj(x))
        out, _ = self.lstm(h)
        pooled = out.mean(dim=1)  # temporal mean pooling
        emb = F.normalize(self.head(pooled), dim=-1)
        return emb  # (B, EMB_DIM)

class Classifier(nn.Module):
    """Optional classifier head for embeddings."""
    def __init__(self, emb_dim=EMB_DIM, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, emb):
        return self.fc(emb)

class GaitNet(nn.Module):
    """Main gait recognition network."""
    def __init__(self, input_dim=INPUT_DIM, emb_dim=EMB_DIM, num_classes=None):
        super().__init__()
        self.encoder = PoseSeqEncoder(input_dim, emb_dim)
        self.classifier = Classifier(emb_dim, num_classes) if num_classes else None

    def forward(self, x):
        emb = self.encoder(x)
        if self.classifier is None:
            return emb
        logits = self.classifier(emb)
        return emb, logits

# ======================
# Normalization Utils
# ======================
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12

def _normalize_seq(seq_np: np.ndarray) -> np.ndarray:
    """
    Normalize (T, 66) sequence:
    - Translate so hip center is origin
    - Scale by shoulder-hip distance
    - Ensure numerical stability
    """
    if seq_np.ndim != 2 or seq_np.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected (T,{INPUT_DIM}) got {seq_np.shape}")

    T = seq_np.shape[0]
    xy = seq_np.reshape(T, NUM_LM, FEAT_PER_LM)  # (T,33,2)

    hip_center = 0.5 * (xy[:, LM_LEFT_HIP, :] + xy[:, LM_RIGHT_HIP, :])
    shoulder_center = 0.5 * (xy[:, LM_LEFT_SHOULDER, :] + xy[:, LM_RIGHT_SHOULDER, :])

    xy_centered = xy - hip_center[:, None, :]
    scale = np.linalg.norm(shoulder_center - hip_center, axis=1)
    scale = np.clip(scale, 1e-3, None)  # avoid division by zero

    xy_norm = xy_centered / scale[:, None, None]
    xy_norm = np.nan_to_num(xy_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return xy_norm.reshape(T, INPUT_DIM).astype(np.float32)

# ======================
# Runtime Wrapper
# ======================
class GaitModel:
    def __init__(self, model_path: str, gallery_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.net = GaitNet()
        self.is_loaded = False

        # Load model weights
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                if isinstance(state, dict) and "state_dict" in state:
                    self.net.load_state_dict(state["state_dict"], strict=False)
                else:
                    self.net.load_state_dict(state, strict=False)
                self.is_loaded = True
                print(f"[GaitModel] Loaded weights from {model_path}")
            except Exception as e:
                print(f"[GaitModel] Failed to load weights: {e}")
        else:
            print(f"[GaitModel] Warning: model file not found at {model_path}")

        self.net.to(self.device).eval()

        # Load gallery (watchlist)
        self.gallery_ids: List[str] = []
        self.gallery_emb: Optional[np.ndarray] = None
        if gallery_path and os.path.exists(gallery_path):
            try:
                data = np.load(gallery_path, allow_pickle=True)
                self.gallery_ids = list(map(str, data["ids"]))
                self.gallery_emb = data["embeddings"].astype(np.float32)
                print(f"[GaitModel] Loaded gallery with {len(self.gallery_ids)} entries")
            except Exception as e:
                print(f"[GaitModel] Failed to load gallery: {e}")

    def gallery_info(self):
        return {"count": len(self.gallery_ids)}

    @torch.no_grad()
    def _embed(self, seq_np: np.ndarray) -> np.ndarray:
        seq_np = _normalize_seq(seq_np)
        x = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)  # (1,T,66)
        emb = self.net.encoder(x).cpu().numpy()[0]
        return emb  # (128,)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    def _behavior_flags(self, seq_np: np.ndarray) -> List[str]:
        """Detect simple suspicious gait patterns."""
        flags = []
        T = seq_np.shape[0]
        xy = seq_np.reshape(T, NUM_LM, FEAT_PER_LM)
        L_ANK, R_ANK = 27, 28
        try:
            lx = xy[:, L_ANK, 0]
            rx = xy[:, R_ANK, 0]
            spread = np.abs(lx - rx)
            if spread.mean() < 0.02:
                flags.append("low_movement")
            dx = np.diff(lx - lx.mean())
            zero_cross = np.sum(np.sign(dx[1:]) != np.sign(dx[:-1]))
            if zero_cross / max(T - 2, 1) < 0.05:
                flags.append("irregular_stride")
        except Exception:
            pass
        return flags

    def infer_and_match(self, seq_np: np.ndarray) -> Dict:
        """Run embedding + optional gallery matching."""
        T = seq_np.shape[0]
        emb = self._embed(seq_np)

        top_k = []
        best_id = None
        best_sim = 0.0

        if self.gallery_emb is not None and len(self.gallery_ids) == self.gallery_emb.shape[0]:
            g_norm = self.gallery_emb / (np.linalg.norm(self.gallery_emb, axis=1, keepdims=True) + 1e-8)
            e_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sims = (g_norm @ e_norm)
            order = np.argsort(-sims)[:5]
            for idx in order:
                top_k.append({"id": self.gallery_ids[idx], "similarity": float(sims[idx])})
            if len(order) > 0:
                best_idx = int(order[0])
                best_id = self.gallery_ids[best_idx]
                best_sim = float(sims[best_idx])

        THRESH = 0.65
        matched = best_sim >= THRESH
        flags = self._behavior_flags(seq_np)

        return {
            "matched": bool(matched),
            "confidence": float(max(best_sim, 0.0)),
            "suspect_id": best_id,
            "similarity": best_sim,
            "top_k": top_k,
            "suspicious_behaviors": flags,
            "frames_used": int(T),
        }
