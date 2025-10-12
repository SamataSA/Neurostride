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
FEAT_PER_LM = 2       # Using x, y only
INPUT_DIM = NUM_LM * FEAT_PER_LM  # 66
EMB_DIM = 128

# ======================
# Network Definitions
# ======================
class PoseSeqEncoder(nn.Module):
    """
    Encodes a sequence of 2D pose keypoints into a robust gait embedding.
    - Uses bidirectional LSTM for temporal context
    - Mean pooling over time
    - L2-normalized embeddings
    """
    def __init__(self, input_dim=INPUT_DIM, emb_dim=EMB_DIM, hidden=192, num_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            hidden, hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor (B, T, INPUT_DIM)
        Returns:
            emb: Tensor (B, EMB_DIM)
        """
        h = self.proj(x)
        out, _ = self.lstm(h)
        pooled = out.mean(dim=1)  # temporal average pooling
        emb = F.normalize(self.head(pooled), dim=-1)
        return emb


class Classifier(nn.Module):
    """Classifier head for gait embeddings."""
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
# Normalization Utilities
# ======================
LM_LEFT_HIP = 23
LM_RIGHT_HIP = 24
LM_LEFT_SHOULDER = 11
LM_RIGHT_SHOULDER = 12

def _normalize_seq(seq_np: np.ndarray) -> np.ndarray:
    """
    Normalize (T, 66) pose sequence:
    - Translate hip center to origin
    - Scale by shoulder–hip distance
    - Ensure stable rotation and zero-mean
    """
    if seq_np.ndim != 2 or seq_np.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected (T, {INPUT_DIM}) but got {seq_np.shape}")

    T = seq_np.shape[0]
    xy = seq_np.reshape(T, NUM_LM, FEAT_PER_LM)

    # Calculate body reference points
    hip_center = 0.5 * (xy[:, LM_LEFT_HIP, :] + xy[:, LM_RIGHT_HIP, :])
    shoulder_center = 0.5 * (xy[:, LM_LEFT_SHOULDER, :] + xy[:, LM_RIGHT_SHOULDER, :])

    # Center the sequence
    xy_centered = xy - hip_center[:, None, :]

    # Scale normalization (shoulder-hip distance)
    scale = np.linalg.norm(shoulder_center - hip_center, axis=1)
    scale = np.clip(scale, 1e-3, None)
    xy_scaled = xy_centered / scale[:, None, None]

    # Optional: rotation normalization (align shoulder axis)
    shoulder_vec = shoulder_center - hip_center
    angles = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
    cos_a, sin_a = np.cos(-angles), np.sin(-angles)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]]).transpose(2, 0, 1)
    xy_rot = np.einsum("tij,tbj->tbi", rot, xy_scaled)

    xy_norm = np.nan_to_num(xy_rot, nan=0.0, posinf=0.0, neginf=0.0)
    return xy_norm.reshape(T, INPUT_DIM).astype(np.float32)


# ======================
# Runtime Wrapper
# ======================
class GaitModel:
    def __init__(self, model_path: str, gallery_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.net = GaitNet()
        self.is_loaded = False

        # Load model weights
        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device)
                state_dict = state.get("state_dict", state)
                self.net.load_state_dict(state_dict, strict=False)
                self.is_loaded = True
                print(f"[GaitModel] ✅ Loaded model weights from {model_path}")
            except Exception as e:
                print(f"[GaitModel] ⚠️ Failed to load weights: {e}")
        else:
            print(f"[GaitModel] ⚠️ Model file not found: {model_path}")

        self.net.to(self.device).eval()

        # Load gallery (watchlist)
        self.gallery_ids: List[str] = []
        self.gallery_emb: Optional[np.ndarray] = None
        if gallery_path and os.path.exists(gallery_path):
            try:
                data = np.load(gallery_path, allow_pickle=True)
                self.gallery_ids = list(map(str, data["ids"]))
                self.gallery_emb = data["embeddings"].astype(np.float32)
                print(f"[GaitModel] 📁 Loaded gallery with {len(self.gallery_ids)} entries")
            except Exception as e:
                print(f"[GaitModel] ⚠️ Failed to load gallery: {e}")

    def gallery_info(self):
        return {"count": len(self.gallery_ids)}

    # ----------------------
    # Embedding extraction
    # ----------------------
    @torch.no_grad()
    def _embed(self, seq_np: np.ndarray) -> np.ndarray:
        seq_np = _normalize_seq(seq_np)
        x = torch.from_numpy(seq_np).unsqueeze(0).to(self.device)  # (1, T, 66)
        emb = self.net.encoder(x).cpu().numpy()[0]
        return emb  # (128,)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))

    # ----------------------
    # Behavioral analysis
    # ----------------------
    def _behavior_flags(self, seq_np: np.ndarray) -> List[str]:
        """Detect potential suspicious gait patterns."""
        flags = []
        T = seq_np.shape[0]
        xy = seq_np.reshape(T, NUM_LM, FEAT_PER_LM)
        L_ANK, R_ANK = 27, 28

        try:
            # Step width (distance between ankles)
            step_width = np.abs(xy[:, L_ANK, 0] - xy[:, R_ANK, 0])
            if step_width.mean() < 0.015:
                flags.append("very_close_steps")

            # Vertical motion (indicates limp / dragging)
            L_Y = xy[:, L_ANK, 1]
            R_Y = xy[:, R_ANK, 1]
            vertical_diff = np.std(L_Y - R_Y)
            if vertical_diff < 0.005:
                flags.append("low_vertical_movement")

            # Stride rhythm check (irregular pace)
            diff = np.diff(L_Y - R_Y)
            zero_cross = np.sum(np.sign(diff[1:]) != np.sign(diff[:-1]))
            if zero_cross / max(T - 2, 1) < 0.05:
                flags.append("irregular_stride")
        except Exception:
            pass

        return flags

    # ----------------------
    # Inference + Matching
    # ----------------------
    def infer_and_match(self, seq_np: np.ndarray) -> Dict:
        """
        Run embedding and optionally compare to gallery.
        Returns structured inference dictionary.
        """
        T = seq_np.shape[0]
        emb = self._embed(seq_np)

        best_id, best_sim = None, 0.0
        top_k = []

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

        # Confidence & threshold
        THRESH = 0.70
        confidence = float(np.clip(best_sim, 0.0, 1.0))
        matched = best_sim >= THRESH

        flags = self._behavior_flags(seq_np)

        return {
            "matched": bool(matched),
            "confidence": confidence,
            "suspect_id": best_id,
            "similarity": best_sim,
            "top_k": top_k,
            "suspicious_behaviors": flags,
            "frames_used": int(T),
        }
