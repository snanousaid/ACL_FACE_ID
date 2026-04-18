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

import face_events
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

# --- Poses disponibles ---
ALL_POSES = ["center", "left", "right", "up", "down"]
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
    if pitch < -0.06 and abs(yaw) < 0.30:
        return "up", yaw, pitch
    if pitch > 0.08 and abs(yaw) < 0.30:
        return "down", yaw, pitch
    return "transition", yaw, pitch


class CameraWorker:
    def __init__(self, cfg: dict):
        self.cfg = cfg

        perf = cfg.get("perf", {})
        self.detect_every = max(1, int(perf.get("detect_every_n", 1)))
        self.detect_scale = float(perf.get("detect_scale", 1.0))
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

        # ROI : le visage doit être centré dans le cadre pour déclencher la reco.
        roi_cfg = cfg.get("roi", {}) or {}
        self.roi_enabled = bool(roi_cfg.get("enabled", True))
        self.roi_x = float(roi_cfg.get("x", 0.25))
        self.roi_y = float(roi_cfg.get("y", 0.15))
        self.roi_w = float(roi_cfg.get("w", 0.50))
        self.roi_h = float(roi_cfg.get("h", 0.70))

        enroll_cfg = cfg.get("enrollment", {})
        self.required_poses: list[str] = enroll_cfg.get("required_poses", ["center", "left", "right"])
        self.optional_poses: list[str] = enroll_cfg.get("optional_poses", ["up", "down"])
        self.active_poses: list[str] = self.required_poses + self.optional_poses

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

        # --- throttling des événements socket (granted/denied) ---
        self._last_event_ts: float = 0.0
        self._last_event_key: str = ""

        # --- pause reconnaissance (pendant la config admin, hors enrôlement) ---
        self._paused = False

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

        # --- vignettes de l'enrôlement, une JPEG par pose ---
        self._pose_thumbs: dict[str, bytes] = {}

        self.logger.info(
            f"STARTUP,camera_worker,os={platform.system()},arch={platform.machine()},"
            f"py={platform.python_version()},{self.actuator.describe()}"
        )
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------- helpers ROI ----------
    def _face_in_roi(self, face_row: np.ndarray, frame_w: int, frame_h: int) -> bool:
        """Retourne True si au moins 70% de la surface du visage est dans la ROI.

        Tolère un léger dépassement (menton, front, joues) : on calcule le
        ratio d'intersection aire(face ∩ ROI) / aire(face) et on compare à 0.70.
        """
        if not self.roi_enabled:
            return True
        x, y, w, h = face_row[:4]
        left = float(x) / frame_w
        top = float(y) / frame_h
        right = float(x + w) / frame_w
        bottom = float(y + h) / frame_h
        roi_right = self.roi_x + self.roi_w
        roi_bottom = self.roi_y + self.roi_h

        inter_w = max(0.0, min(right, roi_right) - max(left, self.roi_x))
        inter_h = max(0.0, min(bottom, roi_bottom) - max(top, self.roi_y))
        inter_area = inter_w * inter_h
        face_area = max(1e-9, (right - left) * (bottom - top))
        return (inter_area / face_area) >= 0.70

    # ---------- helpers enrôlement ----------
    def _enroll_next_pose(self) -> Optional[str]:
        for p in self.required_poses:
            if len(self._enroll_bins.get(p, [])) < self._enroll_target_per_pose:
                return p
        for p in self.optional_poses:
            if len(self._enroll_bins.get(p, [])) < self._enroll_target_per_pose:
                return p
        return None

    def _enroll_all_required_done(self) -> bool:
        return all(len(self._enroll_bins.get(p, [])) >= self._enroll_target_per_pose
                   for p in self.required_poses)

    def _try_sample(self, frame: np.ndarray, face_row: np.ndarray, feat: np.ndarray) -> None:
        """Accepte l'échantillon si : enrôlement actif, dans ROI, qualité OK, pose requise pas pleine."""
        if not self._enroll_active:
            return

        if not self._face_in_roi(face_row, frame.shape[1], frame.shape[0]):
            self._enroll_last_msg = "visage hors du cadre"
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
        if pose == "transition" or pose not in self.active_poses:
            self._enroll_last_msg = "pose intermédiaire"
            return

        bucket = self._enroll_bins.setdefault(pose, [])
        if len(bucket) >= self._enroll_target_per_pose:
            self._enroll_last_msg = f"pose '{POSE_LABELS[pose]}' complète"
            return

        bucket.append(feat)
        self._enroll_last_ts = now
        self._enroll_last_msg = f"+1 {pose} ({len(bucket)}/{self._enroll_target_per_pose})"

        # vignette de la face capturée (crop bbox → 96×96)
        x, y, w, h = face_row[:4].astype(int)
        x = max(x, 0); y = max(y, 0)
        x2 = min(x + w, frame.shape[1]); y2 = min(y + h, frame.shape[0])
        crop = frame[y:y2, x:x2]
        if crop.size > 0:
            thumb = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
            ok_j, buf = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok_j:
                self._pose_thumbs[pose] = buf.tobytes()

    # ---------- boucle principale ----------
    def _loop(self) -> None:
        while not self._stop.is_set():
            jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            if not self.cap.isOpened():
                time.sleep(1.0)
                continue
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            self._frame_idx += 1
            # Pause : on streame la caméra brute, sans détection ni reco.
            # L'enrôlement force toujours la détection (même en pause).
            recognition_on = (not self._paused) or self._enroll_active
            do_detect = recognition_on and (self._frame_idx % self.detect_every == 0)

            bright = mean_brightness(frame)
            lum_ok, lum_msg = brightness_ok(bright, self.cfg)

            if not recognition_on:
                # Flux brut uniquement — pas d'overlays, pas d'events
                state = {
                    "brightness": float(bright), "face": False,
                    "name": None, "role": None, "score": 0.0,
                    "access": "paused", "msg": "pause",
                    "ts": time.time(),
                }
                with self._lock:
                    self._status = state
                self._last_box = None
                self._last_label = ""
                ok_jpg, buf = cv2.imencode(".jpg", frame, jpeg_params)
                with self._lock:
                    if ok_jpg:
                        self._jpeg = buf.tobytes()
                continue

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
                    # détection sur frame réduite pour gagner du CPU ; coords scalées ensuite
                    if self.detect_scale < 0.99:
                        small = cv2.resize(frame, None, fx=self.detect_scale,
                                           fy=self.detect_scale,
                                           interpolation=cv2.INTER_AREA)
                        faces = self.fp.detect(small)
                        if faces is not None and len(faces) > 0:
                            faces = faces.copy()
                            faces[:, :14] *= (1.0 / self.detect_scale)
                    else:
                        faces = self.fp.detect(frame)

                    if faces is not None and len(faces) >= 1:
                        face = FaceProcessor.best_face(faces)
                        state["face"] = True
                        in_roi = self._face_in_roi(face, frame.shape[1], frame.shape[0])
                        state["in_roi"] = in_roi

                        x, y, w, h = face[:4].astype(int)
                        box = (x, y, w, h)

                        if not in_roi:
                            # visage détecté mais le rectangle dépasse le cadre → pas de reco
                            state["access"] = "out_of_zone"
                            state["msg"] = "placez votre visage complètement dans le cadre"
                            label = "CENTRER DANS LE CADRE"
                            color = (160, 160, 160)
                        else:
                            feat = self.fp.embed(frame, face)

                            with self._lock:
                                self._try_sample(frame, face, feat)

                            # Scan sur TOUS les users (actifs + désactivés) pour pouvoir
                            # distinguer "inconnu" (silencieux) de "désactivé" (denied).
                            name, score, entry = match(feat, self.db, include_inactive=True)
                            state["score"] = float(score)
                            active = bool(entry.get("active", True)) if entry else True

                            if name is not None and score >= self.threshold and active:
                                # Utilisateur enregistré et ACTIF → granted
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
                            elif name is not None and score >= self.threshold and not active:
                                # Utilisateur enregistré mais DÉSACTIVÉ → denied (évènement émis)
                                role = entry.get("role", "user") if entry else "user"
                                state["name"] = name
                                state["role"] = role
                                state["access"] = "denied"
                                state["reason"] = "disabled"
                                self.logger.info(f"DENIED_DISABLED,{name},score={score:.3f}")
                                label = f"{name} DÉSACTIVÉ"
                                color = (0, 0, 220)
                            else:
                                # Inconnu (score trop bas) → silencieux, PAS d'événement
                                state["access"] = "unknown"
                                state["name"] = None
                                label = f"Inconnu {score:.2f}"
                                color = (150, 150, 150)
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

                # émission événement socket (throttlé par cooldown)
                if state["access"] in ("granted", "denied") and not self._enroll_active:
                    self._maybe_emit_event(state)
            else:
                # frame sautée — réutilise l'état et la bbox précédents
                state = self._status
                color = self._last_color
                label = self._last_label
                box = self._last_box

            # Overlays UNIQUEMENT pendant l'enrôlement.
            # Sur le home, le flux est propre (pas de bbox, pas de texte) — les infos
            # sont affichées côté frontend (RoiOverlay + AccessCard).
            with self._lock:
                if self._enroll_active:
                    if box is not None:
                        x, y, w, h = box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)
                    next_p = self._enroll_next_pose()
                    hint = f"ENROLL -> {POSE_LABELS.get(next_p, '—')}" if next_p else "ENROLL OK"
                    cv2.putText(frame, hint, (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 200, 255), 2)
                    for i, p in enumerate(self.active_poses):
                        cnt = len(self._enroll_bins.get(p, []))
                        tgt = self._enroll_target_per_pose
                        done = cnt >= tgt
                        req = "!" if p in self.required_poses else "?"
                        txt = f"{p[:1].upper()}{req}:{cnt}/{tgt}"
                        col = (0, 220, 0) if done else (0, 200, 255)
                        cv2.putText(frame, txt, (10 + i * 100, 88),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            ok_jpg, buf = cv2.imencode(".jpg", frame, jpeg_params)
            with self._lock:
                if ok_jpg:
                    self._jpeg = buf.tobytes()

        try:
            self.cap.release()
        except Exception:
            pass

    # ---------- émission événements WebSocket ----------
    def _maybe_emit_event(self, state: dict) -> None:
        """Pousse un AccessEvent sur le bus face_events, throttlé par cooldown."""
        access = state.get("access")
        name = state.get("name")
        key = f"{access}:{name or ''}"
        now = time.time()
        if key == self._last_event_key and (now - self._last_event_ts) < self.cooldown:
            return
        self._last_event_key = key
        self._last_event_ts = now

        granted = access == "granted"
        display_name = name if (name and name != "unknown") else None
        payload = {
            "eventType": "ACCESS_GRANTED" if granted else "ACCESS_DENIED",
            "status": granted,
            "source": "face",
            "score": float(state.get("score", 0.0)),
            "user": {"first_name": display_name} if display_name else None,
            "doorName": "Face ID",
            "readerName": "Caméra locale",
            "createdAt": datetime.now().isoformat(),
        }
        reason = state.get("reason")
        if reason:
            payload["reason"] = reason
        face_events.emit(payload)

    # ---------- API thread-safe ----------
    def pause_recognition(self) -> None:
        with self._lock:
            self._paused = True

    def resume_recognition(self) -> None:
        with self._lock:
            self._paused = False

    def get_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def get_status(self) -> dict:
        with self._lock:
            s = dict(self._status)
            s["enrolling"] = self._enroll_active
            s["roi"] = {
                "enabled": self.roi_enabled,
                "x": self.roi_x,
                "y": self.roi_y,
                "w": self.roi_w,
                "h": self.roi_h,
            }
            if self._enroll_active:
                s["enroll_poses"] = [
                    {
                        "id": p,
                        "label": POSE_LABELS.get(p, p),
                        "count": len(self._enroll_bins.get(p, [])),
                        "target": self._enroll_target_per_pose,
                        "done": len(self._enroll_bins.get(p, [])) >= self._enroll_target_per_pose,
                        "required": p in self.required_poses,
                    }
                    for p in self.active_poses
                ]
                s["enroll_next"] = self._enroll_next_pose()
                s["enroll_current_pose"] = self._enroll_current_pose
                s["enroll_complete"] = self._enroll_all_required_done()
                s["enroll_msg"] = self._enroll_last_msg
                s["enroll_name"] = self._enroll_meta.get("name", "")
        return s

    def start_enroll(self, n_per_pose: int, name: str, role: str) -> tuple[bool, str]:
        with self._lock:
            if self._enroll_active:
                return False, "Un enrôlement est déjà en cours."
            self._enroll_active = True
            self._enroll_bins = {p: [] for p in self.active_poses}
            self._enroll_target_per_pose = max(1, int(n_per_pose))
            self._enroll_meta = {"name": name.strip(), "role": (role.strip() or "user")}
            self._enroll_last_ts = 0.0
            self._enroll_last_msg = ""
            self._pose_thumbs = {}
        n_req = len(self.required_poses)
        n_opt = len(self.optional_poses)
        return True, (
            f"Enrôlement démarré — {n_per_pose}/pose × {n_req} obligatoires + {n_opt} optionnelles."
        )

    def cancel_enroll(self) -> None:
        """Annule SANS sauvegarder. Équivalent à 'rien n'a été pris'."""
        with self._lock:
            self._enroll_active = False
            self._enroll_bins = {}
            self._enroll_target_per_pose = 0
            self._enroll_meta = {}
            self._enroll_last_msg = "annulé — aucun utilisateur créé"
            self._pose_thumbs = {}

    def get_pose_thumb(self, pose: str) -> Optional[bytes]:
        with self._lock:
            return self._pose_thumbs.get(pose)

    def finalize_enroll(self) -> tuple[bool, str]:
        """Sauvegarde uniquement si TOUTES les poses sont complètes. Comme iPhone."""
        with self._lock:
            if not self._enroll_active:
                return False, "Aucun enrôlement en cours."
            meta = dict(self._enroll_meta)
            bins = self._enroll_bins
            target = self._enroll_target_per_pose
            missing = [POSE_LABELS.get(p, p) for p in self.required_poses
                       if len(bins.get(p, [])) < target]

            if missing:
                self._enroll_active = False
                self._enroll_bins = {}
                self._enroll_target_per_pose = 0
                self._enroll_meta = {}
                self._enroll_last_msg = "refusé — poses obligatoires incomplètes"
                self._pose_thumbs = {}
                return False, (
                    "Utilisateur NON enregistré. Poses obligatoires manquantes : "
                    + ", ".join(missing)
                )

            # Toutes les poses obligatoires OK + optionnelles si capturées
            all_feats = []
            for p in self.active_poses:
                all_feats.extend(bins.get(p, []))
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
            self._pose_thumbs = {}
            n_opt = sum(1 for p in self.optional_poses if len(bins.get(p, [])) > 0)
            verb = "mis à jour" if existed else "ajouté"
            return True, (
                f"'{name}' {verb} ({len(all_feats)} échantillons, "
                f"{len(self.required_poses)} obligatoires + {n_opt} bonus)."
            )

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
