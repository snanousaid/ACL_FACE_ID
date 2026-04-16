"""Worker caméra singleton: capture + détection + reco + stream + enrôlement multi-poses.

Enrôlement iPhone-like: exige N échantillons pour chacune des 5 poses
(centre/gauche/droite/haut/bas). finalize() refuse si une pose est vide.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from gpio import AccessActuator
from utils import (
    FaceProcessor,
    brightness_ok,
    load_known,
    match,
    mean_brightness,
    open_camera,
    save_known,
)

# --- Poses requises (équivalent cercle iPhone Face ID) ---
POSE_ORDER = ["center", "left", "right", "up", "down"]
POSE_LABELS = {
    "center": "Face caméra",
    "left":   "Tourner à gauche",
    "right":  "Tourner à droite",
    "up":     "Lever le menton",
    "down":   "Baisser le menton",
}

# Qualité minimale par échantillon
MIN_FACE_SCORE = 0.80
MIN_FACE_WIDTH = 80         # pixels
MIN_SAMPLE_INTERVAL = 0.15  # s — rate-limit


def _setup_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("access")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = RotatingFileHandler(log_path, maxBytes=512_000, backupCount=5, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s,%(levelname)s,%(message)s"))
    logger.addHandler(h)
    return logger


def _estimate_pose(face_row: np.ndarray) -> tuple[str, float, float]:
    """Retourne (bin, yaw, pitch) à partir des 5 landmarks YuNet.

    YuNet ordre: [x,y,w,h, rex,rey, lex,ley, nx,ny, rmx,rmy, lmx,lmy, score]
    (right_eye = œil droit du sujet = côté gauche de l'image)
    """
    x, y, w, h = face_row[:4]
    rex, rey = face_row[4], face_row[5]
    lex, ley = face_row[6], face_row[7]
    nx, ny   = face_row[8], face_row[9]
    rmx, rmy = face_row[10], face_row[11]
    lmx, lmy = face_row[12], face_row[13]

    eye_cx = (rex + lex) / 2.0
    eye_cy = (rey + ley) / 2.0
    mouth_cy = (rmy + lmy) / 2.0
    inter_eye = max(abs(lex - rex), 1.0)

    # yaw: décalage horizontal du nez / distance inter-oculaire
    yaw = (nx - eye_cx) / inter_eye

    # pitch: ratio vertical nez entre yeux (0) et bouche (1), centré sur 0.5
    em = mouth_cy - eye_cy
    pitch = ((ny - eye_cy) / em - 0.5) if em > 1 else 0.0

    # Classification
    if abs(yaw) < 0.18 and abs(pitch) < 0.07:
        return "center", yaw, pitch
    if yaw < -0.30 and abs(pitch) < 0.20:
        return "left", yaw, pitch
    if yaw > 0.30 and abs(pitch) < 0.20:
        return "right", yaw, pitch
    if pitch < -0.10 and abs(yaw) < 0.25:
        return "up", yaw, pitch
    if pitch > 0.12 and abs(yaw) < 0.25:
        return "down", yaw, pitch
    return "transition", yaw, pitch


class CameraWorker:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        perf = cfg.get("perf", {})
        self.detect_every = max(1, int(perf.get("detect_every_n", 1)))
        self.jpeg_quality = int(perf.get("jpeg_quality", 70))
        n_threads = int(perf.get("opencv_threads", 0))
        if n_threads > 0:
            cv2.setNumThreads(n_threads)

        self.fp = FaceProcessor(cfg)
        self.cap = open_camera(cfg)
        self.db = load_known(cfg["paths"]["embeddings"])
        self.logger = _setup_logger(cfg["paths"]["log_file"])

        self.threshold = float(cfg["recognition"]["match_threshold"])
        self.cooldown = float(cfg["access"]["cooldown_seconds"])
        self.unlock_s = int(cfg["access"]["unlock_seconds"])

        gpio_cfg = cfg.get("gpio", {})
        self.actuator = AccessActuator(
            chip=gpio_cfg.get("chip", "/dev/gpiochip1"),
            line=int(gpio_cfg.get("line", 7)),
            active_high=bool(gpio_cfg.get("active_high", True)),
            enabled=bool(gpio_cfg.get("enabled", True)),
        )

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._jpeg: Optional[bytes] = None
        self._status: dict = {
            "name": None, "role": None, "score": 0.0,
            "access": "waiting", "brightness": 0.0, "face": False,
            "ts": 0.0, "msg": "démarrage…",
        }
        self._last_grant: dict[str, float] = {}

        # --- état enrôlement multi-poses ---
        self._enroll_active = False
        self._enroll_meta: dict = {}
        self._enroll_bins: dict[str, list[np.ndarray]] = {}
        self._enroll_target_per_pose: int = 0
        self._enroll_last_ts: float = 0.0
        self._enroll_current_pose: str = "transition"
        self._enroll_last_msg: str = ""

        # --- cache overlay pour frames sautées (skip-frame) ---
        self._frame_idx: int = 0
        self._last_box: Optional[tuple[int, int, int, int]] = None
        self._last_label: str = ""
        self._last_color: tuple[int, int, int] = (200, 200, 200)

        self.logger.info(
            f"STARTUP,camera_worker,os={platform.system()},arch={platform.machine()},"
            f"py={platform.python_version()},{self.actuator.describe()}"
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------- helpers enrôlement ----------
    def _enroll_next_pose(self) -> Optional[str]:
        for p in POSE_ORDER:
            if len(self._enroll_bins.get(p, [])) < self._enroll_target_per_pose:
                return p
        return None

    def _enroll_all_done(self) -> bool:
        return all(len(self._enroll_bins.get(p, [])) >= self._enroll_target_per_pose
                   for p in POSE_ORDER)

    def _try_sample(self, face_row: np.ndarray, feat: np.ndarray) -> None:
        """Accepte l'échantillon si : enrôlement actif, qualité OK, pose requise pas pleine."""
        if not self._enroll_active:
            return

        score = float(face_row[-1])
        face_w = float(face_row[2])
        if score < MIN_FACE_SCORE or face_w < MIN_FACE_WIDTH:
            self._enroll_last_msg = f"qualité faible (score={score:.2f}, w={face_w:.0f})"
            return

        now = time.time()
        if now - self._enroll_last_ts < MIN_SAMPLE_INTERVAL:
            return

        pose, _yaw, _pitch = _estimate_pose(face_row)
        self._enroll_current_pose = pose
        if pose == "transition":
            self._enroll_last_msg = "pose intermédiaire"
            return

        bucket = self._enroll_bins.setdefault(pose, [])
        if len(bucket) >= self._enroll_target_per_pose:
            self._enroll_last_msg = f"pose '{POSE_LABELS[pose]}' complète"
            return

        bucket.append(feat)
        self._enroll_last_ts = now
        self._enroll_last_msg = f"+1 {pose} ({len(bucket)}/{self._enroll_target_per_pose})"

    # ---------- boucle principale ----------
    def _loop(self) -> None:
        while not self._stop.is_set():
            jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            self._frame_idx += 1
            do_detect = (self._frame_idx % self.detect_every == 0)

            bright = mean_brightness(frame)
            lum_ok, lum_msg = brightness_ok(bright, self.cfg)

            if do_detect:
                state = {
                    "brightness": float(bright), "face": False,
                    "name": None, "role": None, "score": 0.0,
                    "access": "waiting", "msg": lum_msg,
                    "ts": time.time(),
                }
                color = (200, 200, 200)
                label = lum_msg
                box: Optional[tuple[int, int, int, int]] = None

                if lum_ok:
                    faces = self.fp.detect(frame)
                    if faces is not None and len(faces) >= 1:
                        face = FaceProcessor.best_face(faces)
                        feat = self.fp.embed(frame, face)
                        state["face"] = True

                        with self._lock:
                            self._try_sample(face, feat)

                        name, score, entry = match(feat, self.db)
                        state["score"] = float(score)

                        if name is not None and score >= self.threshold:
                            role = entry.get("role", "user") if entry else "user"
                            state["name"] = name
                            state["role"] = role
                            state["access"] = "granted"
                            now = time.time()
                            if now - self._last_grant.get(name, 0.0) >= self.cooldown:
                                self.logger.info(f"GRANTED,{name},role={role},score={score:.3f}")
                                self._last_grant[name] = now
                                self.actuator.pulse(self.unlock_s)
                            label = f"{name} [{role}] {score:.2f}"
                            color = (0, 220, 0)
                        else:
                            state["access"] = "denied"
                            state["name"] = name or "unknown"
                            self.logger.info(f"DENIED,{name or 'unknown'},score={score:.3f}")
                            label = f"DENIED {score:.2f}"
                            color = (0, 0, 220)

                        x, y, w, h = face[:4].astype(int)
                        box = (x, y, w, h)
                    else:
                        state["msg"] = "no_face"
                        label = "no face"
                else:
                    color = (0, 0, 220)
                    self.logger.warning(f"brightness,{lum_msg}")

                # mise à jour état + cache pour frames sautées suivantes
                with self._lock:
                    self._status = state
                self._last_box = box
                self._last_label = label
                self._last_color = color
            else:
                # frame sautée — réutilise l'état et la bbox précédents
                state = self._status
                color = self._last_color
                label = self._last_label
                box = self._last_box

            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"lum:{bright:.0f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # overlay enrôlement
            with self._lock:
                if self._enroll_active:
                    next_p = self._enroll_next_pose()
                    hint = f"ENROLL -> {POSE_LABELS.get(next_p, '—')}" if next_p else "ENROLL OK"
                    cv2.putText(frame, hint, (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 200, 255), 2)
                    # mini tableau de bord
                    for i, p in enumerate(POSE_ORDER):
                        cnt = len(self._enroll_bins.get(p, []))
                        tgt = self._enroll_target_per_pose
                        done = cnt >= tgt
                        txt = f"{p[:1].upper()}:{cnt}/{tgt}"
                        col = (0, 220, 0) if done else (0, 200, 255)
                        cv2.putText(frame, txt, (10 + i * 90, 88),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

            ok_jpg, buf = cv2.imencode(".jpg", frame, jpeg_params)
            with self._lock:
                if ok_jpg:
                    self._jpeg = buf.tobytes()

        try:
            self.cap.release()
        except Exception:
            pass

    # ---------- API thread-safe ----------
    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def get_status(self) -> dict:
        with self._lock:
            s = dict(self._status)
            s["enrolling"] = self._enroll_active
            if self._enroll_active:
                s["enroll_poses"] = [
                    {
                        "id": p,
                        "label": POSE_LABELS[p],
                        "count": len(self._enroll_bins.get(p, [])),
                        "target": self._enroll_target_per_pose,
                        "done": len(self._enroll_bins.get(p, [])) >= self._enroll_target_per_pose,
                    }
                    for p in POSE_ORDER
                ]
                s["enroll_next"] = self._enroll_next_pose()
                s["enroll_current_pose"] = self._enroll_current_pose
                s["enroll_complete"] = self._enroll_all_done()
                s["enroll_msg"] = self._enroll_last_msg
                s["enroll_name"] = self._enroll_meta.get("name", "")
        return s

    def start_enroll(self, n_per_pose: int, name: str, role: str) -> tuple[bool, str]:
        with self._lock:
            if self._enroll_active:
                return False, "Un enrôlement est déjà en cours."
            self._enroll_active = True
            self._enroll_bins = {p: [] for p in POSE_ORDER}
            self._enroll_target_per_pose = max(1, int(n_per_pose))
            self._enroll_meta = {"name": name.strip(), "role": (role.strip() or "user")}
            self._enroll_last_ts = 0.0
            self._enroll_last_msg = ""
        return True, (
            f"Enrôlement démarré — {n_per_pose} échantillons × {len(POSE_ORDER)} poses."
        )

    def cancel_enroll(self) -> None:
        """Annule SANS sauvegarder. Équivalent à 'rien n'a été pris'."""
        with self._lock:
            self._enroll_active = False
            self._enroll_bins = {}
            self._enroll_target_per_pose = 0
            self._enroll_meta = {}
            self._enroll_last_msg = "annulé — aucun utilisateur créé"

    def finalize_enroll(self) -> tuple[bool, str]:
        """Sauvegarde uniquement si TOUTES les poses sont complètes. Comme iPhone."""
        with self._lock:
            if not self._enroll_active:
                return False, "Aucun enrôlement en cours."
            meta = dict(self._enroll_meta)
            bins = self._enroll_bins
            target = self._enroll_target_per_pose
            missing = [POSE_LABELS[p] for p in POSE_ORDER
                       if len(bins.get(p, [])) < target]

            if missing:
                # NE PAS sauvegarder — on relâche l'état et on retourne l'erreur
                self._enroll_active = False
                self._enroll_bins = {}
                self._enroll_target_per_pose = 0
                self._enroll_meta = {}
                self._enroll_last_msg = "refusé — poses incomplètes"
                return False, (
                    "Utilisateur NON enregistré. Poses incomplètes : "
                    + ", ".join(missing)
                )

            # Toutes les poses OK → moyenne globale des embeddings
            all_feats = []
            for p in POSE_ORDER:
                all_feats.extend(bins[p])
            mean_feat = np.mean(np.stack(all_feats), axis=0).astype(np.float32)

            name = meta.get("name", "")
            if not name:
                self._enroll_active = False
                return False, "Nom manquant."

            db = load_known(self.cfg["paths"]["embeddings"])
            existed = name in db
            db[name] = {
                "embedding": mean_feat,
                "role": meta.get("role", "user"),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "active": True,
            }
            save_known(self.cfg["paths"]["embeddings"], db)
            self.db = db

            # reset
            self._enroll_active = False
            self._enroll_bins = {}
            self._enroll_target_per_pose = 0
            self._enroll_meta = {}
            self._enroll_last_msg = ""
            verb = "mis à jour" if existed else "ajouté"
            return True, f"'{name}' {verb} ({len(all_feats)} échantillons, 5 poses)."

    def reload_db(self) -> None:
        db = load_known(self.cfg["paths"]["embeddings"])
        with self._lock:
            self.db = db

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.actuator.close()
        except Exception:
            pass
