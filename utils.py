"""Shared helpers: camera init, face detection/embedding, matching, brightness."""
from __future__ import annotations

import os
import pickle
import platform
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pick_backend(name: str) -> int:
    name = (name or "AUTO").upper()
    if name == "AUTO":
        if platform.system() == "Linux":
            return cv2.CAP_V4L2
        if platform.system() == "Windows":
            return cv2.CAP_DSHOW
        return cv2.CAP_ANY
    return {
        "V4L2": cv2.CAP_V4L2,
        "DSHOW": cv2.CAP_DSHOW,
        "MSMF": cv2.CAP_MSMF,
        "ANY": cv2.CAP_ANY,
    }.get(name, cv2.CAP_ANY)


def open_camera(cfg: dict) -> cv2.VideoCapture:
    cam_cfg = cfg["camera"]
    idx = cam_cfg["index"]
    backend = _pick_backend(cam_cfg.get("backend", "AUTO"))

    # Tentative 1 : index demandé + backend préféré
    cap = cv2.VideoCapture(idx, backend)

    # Tentative 2 : index demandé, backend auto
    if not cap.isOpened():
        print(f"[WARN] Camera {idx} (backend={backend}) échouée — essai backend AUTO")
        cap = cv2.VideoCapture(idx, cv2.CAP_ANY)

    # Tentative 3 : scanner les index 0-4
    if not cap.isOpened():
        for try_idx in range(5):
            if try_idx == idx:
                continue
            print(f"[WARN] Essai /dev/video{try_idx}…")
            cap = cv2.VideoCapture(try_idx, cv2.CAP_ANY)
            if cap.isOpened():
                print(f"[OK] Caméra trouvée sur index {try_idx}")
                break

    # Tentative 4 : chemin direct V4L2 (Linux)
    if not cap.isOpened() and platform.system() == "Linux":
        for dev in ("/dev/video0", "/dev/video1", "/dev/video2", "/dev/video4"):
            print(f"[WARN] Essai {dev}…")
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if cap.isOpened():
                print(f"[OK] Caméra trouvée sur {dev}")
                break

    if not cap.isOpened():
        print("[ERR] Aucune caméra trouvée. Vérifier :")
        print("      - ls /dev/video*")
        print("      - sudo usermod -aG video $USER  (puis relogger)")
        print("      - v4l2-ctl --list-devices")
        print("[WARN] Démarrage SANS caméra — stream indisponible.")
        return cap  # retourne un cap non-ouvert au lieu de crasher

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def mean_brightness(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def brightness_ok(value: float, cfg: dict) -> tuple[bool, str]:
    b = cfg["brightness"]
    if value < b["min_mean"]:
        return False, f"too_dark({value:.1f})"
    if value > b["max_mean"]:
        return False, f"overexposed({value:.1f})"
    return True, "ok"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


class FaceProcessor:
    """Détection YuNet + embedding SFace (OpenCV, ONNX) — cross-platform."""

    def __init__(self, cfg: dict):
        det_path = Path(cfg["detection"]["model"])
        rec_path = Path(cfg["recognition"]["model"])
        for p in (det_path, rec_path):
            if not p.exists():
                print(f"[ERR] Modèle manquant: {p}", file=sys.stderr)
                print("      Voir les URLs en tête de config.yaml", file=sys.stderr)
                sys.exit(2)

        self.input_size = (cfg["camera"]["width"], cfg["camera"]["height"])
        self.detector = cv2.FaceDetectorYN_create(
            str(det_path),
            "",
            self.input_size,
            cfg["detection"]["score_threshold"],
            cfg["detection"]["nms_threshold"],
            cfg["detection"]["top_k"],
        )
        self.recognizer = cv2.FaceRecognizerSF_create(str(rec_path), "")

    def detect(self, frame: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        if (w, h) != self.input_size:
            self.input_size = (w, h)
            self.detector.setInputSize(self.input_size)
        _, faces = self.detector.detect(frame)
        return faces  # None ou Nx15 (bbox+5 landmarks+score)

    def embed(self, frame: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame, face_row)
        feat = self.recognizer.feature(aligned)
        return feat.flatten().astype(np.float32)

    @staticmethod
    def best_face(faces: np.ndarray) -> np.ndarray:
        # le plus grand visage (surface bbox)
        areas = faces[:, 2] * faces[:, 3]
        return faces[int(np.argmax(areas))]


def _as_entry(value, name: str = "") -> dict:
    """Rétrocompat: ancien format = np.ndarray brut; nouveau = dict avec méta."""
    if isinstance(value, dict) and "embedding" in value:
        return value
    return {
        "embedding": np.asarray(value, dtype=np.float32),
        "role": "user",
        "created_at": "",
        "active": True,
    }


def load_known(path: str) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            raw = pickle.load(f)
    except (ModuleNotFoundError, Exception) as e:
        print(f"[WARN] Impossible de charger {p}: {e}")
        print(f"[WARN] Fichier créé avec une version numpy incompatible. "
              f"Supprime-le et ré-enrôle: rm {p}")
        return {}
    return {name: _as_entry(v, name) for name, v in raw.items()}


def save_known(path: str, db: dict[str, dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)


def match(embedding: np.ndarray, db: dict[str, dict]) -> tuple[Optional[str], float, Optional[dict]]:
    if not db:
        return None, 0.0, None
    best_name, best_score, best_entry = None, -1.0, None
    for name, entry in db.items():
        if not entry.get("active", True):
            continue
        ref = entry["embedding"]
        s = cosine_similarity(embedding, ref)
        if s > best_score:
            best_name, best_score, best_entry = name, s, entry
    return best_name, best_score, best_entry


def ensure_dirs(cfg: dict) -> None:
    for k in ("embeddings", "log_file"):
        Path(cfg["paths"][k]).parent.mkdir(parents=True, exist_ok=True)
    os.makedirs("models", exist_ok=True)
