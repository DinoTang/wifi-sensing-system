"""Real-time / Offline CSI Pose and Presence Inference Engine (Person 3).

Designed for seamless integration with Backend (FastAPI/WebSocket) and Frontend.
"""

from __future__ import annotations

from collections import deque
import argparse
import json
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from .dataset import CSINormalizer
    from .models import load_checkpoint
except ImportError:
    from dataset import CSINormalizer
    from models import load_checkpoint

# Standard MediaPipe 33 BlazePose landmark names
LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]


class CSIPosePredictor:
    """Production inference engine for CSI Human Presence & 2D Pose Estimation."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "models/checkpoints/best_model.pt",
        normalizer_path: Union[str, Path] = "models/checkpoints/normalization_params.json",
        device: str = "cpu",
        presence_threshold: float = 0.5,
    ):
        self.device = device
        self.presence_threshold = presence_threshold

        self.model, self.config = load_checkpoint(checkpoint_path, device=device)
        self.normalizer = CSINormalizer.load(normalizer_path)

        data_cfg = self.config.get("data", {})
        self.window_size = data_cfg.get("window_size", 20)
        self.subcarriers = data_cfg.get("num_subcarriers", 224)
        self.frame_width = data_cfg.get("frame_width", 1920.0)
        self.frame_height = data_cfg.get("frame_height", 1080.0)
        self.version = self.config.get("version", "1.0.0")

    def predict_window(
        self,
        csi_window: Optional[np.ndarray],
        timestamp: Optional[float] = None,
        max_age_sec: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform inference on a single CSI window.
        
        Args:
            csi_window: Array of shape (window_size=20, subcarriers=224)
            timestamp: Epoch timestamp of the latest frame (seconds or ms)
            max_age_sec: Optional staleness threshold (seconds). If exceeded, returns STALE_INPUT error.
            current_time: Reference wall-clock epoch seconds (defaults to time.time()).
            
        Returns:
            Structured dictionary matching Backend/Frontend contract.
        """
        t0 = time.perf_counter()
        now_sec = current_time if current_time is not None else time.time()
        resolved_ts = timestamp if timestamp is not None else now_sec

        def _error_payload(code: str, msg: str) -> Dict[str, Any]:
            return {
                "status": "error",
                "error_code": code,
                "message": msg,
                "timestamp": resolved_ts,
                "person_present": False,
                "presence_probability": 0.0,
                "keypoints": [],
                "num_keypoints": 0,
                "model_version": self.version,
            }

        # 1. Validation: missing data, staleness, shape, numerical stability
        if csi_window is None:
            return _error_payload("MISSING_DATA", "CSI window input is None")

        if max_age_sec is not None and timestamp is not None:
            ts_sec = timestamp / 1000.0 if timestamp > 1e11 else timestamp
            age_sec = now_sec - ts_sec
            if age_sec > max_age_sec:
                return _error_payload(
                    "STALE_INPUT",
                    f"CSI window timestamp expired (age={age_sec:.2f}s > limit={max_age_sec:.2f}s)",
                )

        if not isinstance(csi_window, np.ndarray):
            csi_window = np.asarray(csi_window, dtype=np.float32)

        if csi_window.shape != (self.window_size, self.subcarriers):
            return _error_payload(
                "INVALID_SHAPE",
                f"Expected shape ({self.window_size}, {self.subcarriers}), got {csi_window.shape}",
            )

        if np.isnan(csi_window).any() or np.isinf(csi_window).any():
            return _error_payload(
                "NUMERICAL_INSTABILITY",
                "CSI window contains NaN or Inf values",
            )

        # 2. Preprocess & Normalize
        csi_norm = self.normalizer.transform(csi_window[None, :, :])  # (1, 20, 224)
        tensor = torch.tensor(csi_norm, dtype=torch.float32, device=self.device)

        # 3. Model Forward Pass
        self.model.eval()
        with torch.no_grad():
            pres_logits, pred_kpts = self.model(tensor)
            pres_prob = float(torch.sigmoid(pres_logits)[0].item())
            kpts_norm = pred_kpts[0].cpu().numpy()  # (33, 2) in [0, 1] range

        is_present = pres_prob >= self.presence_threshold
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # 4. Construct Keypoints Payload
        keypoints_payload = []
        for i, name in enumerate(LANDMARK_NAMES):
            xn = float(kpts_norm[i, 0])
            yn = float(kpts_norm[i, 1])
            px = int(round(xn * self.frame_width))
            py = int(round(yn * self.frame_height))

            # Confidence derived strictly from model presence sigmoid probability (never video label visibility)
            conf = pres_prob if is_present else 0.0

            keypoints_payload.append({
                "id": i,
                "name": name,
                "x_pixel": px,
                "y_pixel": py,
                "x_norm": round(xn, 4),
                "y_norm": round(yn, 4),
                "confidence": round(conf, 4),
            })

        return {
            "status": "success",
            "timestamp": resolved_ts,
            "person_present": is_present,
            "presence_probability": round(pres_prob, 4),
            "confidence_method": "model_presence_sigmoid_probability",
            "keypoints": keypoints_payload if is_present else [],
            "num_keypoints": len(LANDMARK_NAMES) if is_present else 0,
            "latency_ms": round(latency_ms, 2),
            "coordinate_system": {
                "type": "2D_pixel",
                "reference_width": self.frame_width,
                "reference_height": self.frame_height,
                "origin": "top_left",
            },
            "model_version": self.version,
        }


class CSIStreamBuffer:
    """Strictly causal sliding-window buffer for real-time streaming CSI frames."""

    def __init__(
        self,
        predictor: Optional[CSIPosePredictor] = None,
        window_size: int = 20,
        step_size: int = 2,
        feature_dim: int = 224,
        max_age_sec: Optional[float] = 5.0,
    ):
        self.predictor = predictor
        self.window_size = window_size
        self.step_size = step_size
        self.feature_dim = predictor.subcarriers if predictor is not None else feature_dim
        self.max_age_sec = max_age_sec
        self.buffer: Deque[np.ndarray] = deque(maxlen=window_size)
        self.timestamps: Deque[float] = deque(maxlen=window_size)
        self._frames_received: int = 0

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.window_size

    def push(self, csi_frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        arr = np.asarray(csi_frame, dtype=np.float32)
        self.buffer.append(arr)
        self.timestamps.append(timestamp if timestamp is not None else time.time())
        self._frames_received += 1

    def get_window(self) -> np.ndarray:
        if not self.is_ready():
            raise RuntimeError(f"Buffer holds {len(self.buffer)}/{self.window_size} frames")
        return np.stack(list(self.buffer), axis=0).astype(np.float32)

    def push_frame(self, csi_frame: np.ndarray, timestamp: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Push a single CSI frame (224,). Emits prediction at window_size and every step_size frames."""
        if csi_frame is None or np.asarray(csi_frame).shape != (self.feature_dim,):
            return {
                "status": "error",
                "error_code": "INVALID_FRAME_SHAPE",
                "message": f"Expected frame ({self.feature_dim},)",
                "person_present": False,
                "keypoints": [],
                "timestamp": timestamp or time.time(),
            }

        self.push(csi_frame, timestamp=timestamp)
        if not self.is_ready():
            return None

        frames_after_warmup = self._frames_received - self.window_size
        if frames_after_warmup % self.step_size != 0:
            return None

        if self.predictor is None:
            return None

        window_arr = self.get_window()
        ts = self.timestamps[-1]
        return self.predictor.predict_window(window_arr, timestamp=ts, max_age_sec=self.max_age_sec)


def main():
    parser = argparse.ArgumentParser(description="Run CSI Pose Inference on Sample Data")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt")
    parser.add_argument("--normalizer", default="models/checkpoints/normalization_params.json")
    parser.add_argument("--sample-npz", default="../signal-processing/data/pose/windows/walking_01.npz")
    args = parser.parse_args()

    predictor = CSIPosePredictor(
        checkpoint_path=args.checkpoint,
        normalizer_path=args.normalizer,
    )

    sample_path = Path(args.sample_npz)
    if not sample_path.exists():
        sample_path = Path("signal-processing/data/pose/windows/walking_01.npz")

    if sample_path.exists():
        with np.load(sample_path, allow_pickle=True) as d:
            csi_samples = d["csi"]
            activity = str(d["activity"])

        print(f"Testing offline inference on {sample_path.name} (Activity: {activity})...")
        sample_win = csi_samples[0]
        res = predictor.predict_window(sample_win, timestamp=time.time(), max_age_sec=5.0)
        print("\n=== Prediction Output (First 5 keypoints preview) ===")
        print(f"Status: {res['status']} | Version: {res['model_version']}")
        print(f"Person Present: {res['person_present']} (Prob: {res['presence_probability']})")
        print(f"Coordinate System: {res['coordinate_system']} | Latency: {res['latency_ms']} ms")
        print("Keypoints snippet:")
        for k in res["keypoints"][:5]:
            print(f"  - {k['name']:<18}: ({k['x_pixel']}px, {k['y_pixel']}px) | norm=({k['x_norm']}, {k['y_norm']}) | conf={k['confidence']}")

        # Causal streaming buffer test on frames from first window + next few frames
        stream_buf = CSIStreamBuffer(predictor=predictor, window_size=20, step_size=2, max_age_sec=None)
        emitted = 0
        for t in range(20):
            out = stream_buf.push_frame(csi_samples[0, t], timestamp=float(t) * 0.1)
            if out is not None and out.get("status") == "success":
                emitted += 1
        print(f"Stream buffer causal test (20 frames pushed): emitted={emitted} prediction(s).")
    else:
        print("Sample NPZ not found, testing dummy array...")
        dummy = np.random.randn(20, 224).astype(np.float32)
        res = predictor.predict_window(dummy)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
