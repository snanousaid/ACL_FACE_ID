"""Contrôle d'accès par reconnaissance faciale.

Pipeline: V4L2/DSHOW → YuNet (détection) → SFace (embedding) → cosine match → décision.
GPIO désactivé pour test Windows; à réactiver pour déploiement A133.
"""
from __future__ import annotations

import csv
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2

from utils import (
    FaceProcessor,
    brightness_ok,
    ensure_dirs,
    load_config,
    load_known,
    match,
    mean_brightness,
    open_camera,
)


def setup_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("access")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    handler = RotatingFileHandler(log_path, maxBytes=512_000, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s,%(levelname)s,%(message)s"))
    logger.addHandler(handler)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(console)
    return logger


def grant_access(name: str, role: str, unlock_seconds: int, logger: logging.Logger) -> None:
    """Stub d'actionneur — à remplacer par pilotage GPIO sur A133.

    Exemple gpiod (déploiement):
        import gpiod
        chip = gpiod.Chip('/dev/gpiochip1')
        line = chip.get_line(7)
        line.request(consumer='access', type=gpiod.LINE_REQ_DIR_OUT)
        line.set_value(1); time.sleep(unlock_seconds); line.set_value(0)
    """
    print(f">>> ACCESS GRANTED : {name} [{role}] (gâche ouverte {unlock_seconds}s)")
    logger.info(f"GRANTED,{name},role={role},unlock_{unlock_seconds}s")


def deny_access(reason: str, score: float, logger: logging.Logger) -> None:
    print(f"--- ACCESS DENIED : {reason} (score={score:.3f})")
    logger.info(f"DENIED,{reason},score={score:.3f}")


def run(cfg: dict) -> int:
    logger = setup_logger(cfg["paths"]["log_file"])
    logger.info("STARTUP,access_control")

    fp = FaceProcessor(cfg)
    db = load_known(cfg["paths"]["embeddings"])
    if not db:
        logger.warning("empty_db,enroll_first — lance: python enroll.py <nom>")
    else:
        logger.info(f"db_loaded,{len(db)}_persons")

    cap = open_camera(cfg)
    threshold = float(cfg["recognition"]["match_threshold"])
    cooldown = float(cfg["access"]["cooldown_seconds"])
    unlock_s = int(cfg["access"]["unlock_seconds"])
    show = bool(cfg["display"]["show_preview"])
    warn_every = int(cfg["brightness"]["warn_every_n_frames"])

    last_grant: dict[str, float] = {}
    lum_warn_counter = 0
    last_bad_lum = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.error("frame_read_failed")
                time.sleep(0.1)
                continue

            bright = mean_brightness(frame)
            lum_ok, lum_msg = brightness_ok(bright, cfg)

            status_text = ""
            color = (0, 255, 255)

            if not lum_ok:
                lum_warn_counter += 1
                if lum_msg != last_bad_lum or lum_warn_counter % warn_every == 0:
                    logger.warning(f"brightness,{lum_msg}")
                    last_bad_lum = lum_msg
                status_text = f"LUM: {lum_msg}"
                color = (0, 0, 255)
            else:
                lum_warn_counter = 0
                last_bad_lum = ""
                faces = fp.detect(frame)
                if faces is None or len(faces) == 0:
                    status_text = "no_face"
                else:
                    face = FaceProcessor.best_face(faces)
                    feat = fp.embed(frame, face)
                    name, score, entry = match(feat, db)

                    x, y, w, h = face[:4].astype(int)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    if name is not None and score >= threshold:
                        role = entry.get("role", "user") if entry else "user"
                        now = time.time()
                        if now - last_grant.get(name, 0.0) >= cooldown:
                            grant_access(name, role, unlock_s, logger)
                            last_grant[name] = now
                        status_text = f"{name} [{role}] ({score:.2f})"
                        color = (0, 255, 0)
                    else:
                        deny_access(name or "unknown", score, logger)
                        status_text = f"DENIED ({score:.2f})"
                        color = (0, 0, 255)

            if show:
                cv2.putText(frame, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"lum:{bright:.0f}", (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("Access Control (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("SHUTDOWN,ctrl_c")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    ensure_dirs(cfg)
    sys.exit(run(cfg))
