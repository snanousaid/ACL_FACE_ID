"""Enrôlement: capture N frames d'un visage, moyenne les embeddings, sauvegarde sous un nom.

Usage:
    python enroll.py <nom> [--role admin|user] [--samples 10]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from utils import (
    FaceProcessor,
    brightness_ok,
    ensure_dirs,
    load_config,
    load_known,
    mean_brightness,
    open_camera,
    save_known,
)


def capture_embeddings(cap, fp: FaceProcessor, cfg: dict, n_samples: int) -> np.ndarray:
    feats: list[np.ndarray] = []
    stable_frames = 0
    print(f"[INFO] Regardez la caméra. Collecte de {n_samples} échantillons...")
    while len(feats) < n_samples:
        ok, frame = cap.read()
        if not ok:
            continue

        b = mean_brightness(frame)
        lum_ok, lum_msg = brightness_ok(b, cfg)
        faces = fp.detect(frame) if lum_ok else None

        if faces is not None and len(faces) >= 1 and lum_ok:
            face = FaceProcessor.best_face(faces)
            feat = fp.embed(frame, face)
            feats.append(feat)
            stable_frames += 1
            x, y, w, h = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{len(feats)}/{n_samples}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            status = lum_msg if not lum_ok else "no_face"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("Enrollment (q to abort)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt
        time.sleep(0.05)

    return np.mean(np.stack(feats), axis=0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="Nom/ID de la personne à enregistrer")
    ap.add_argument("--role", default="user", help="Rôle: user|admin|visitor")
    ap.add_argument("--samples", type=int, default=10, help="Nb d'échantillons à moyenner")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    fp = FaceProcessor(cfg)
    cap = open_camera(cfg)

    try:
        feat = capture_embeddings(cap, fp, cfg, args.samples)
    except KeyboardInterrupt:
        print("[WARN] Enrôlement annulé.")
        return 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    db = load_known(cfg["paths"]["embeddings"])
    overwrite = args.name in db
    db[args.name] = {
        "embedding": feat,
        "role": args.role,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active": True,
    }
    save_known(cfg["paths"]["embeddings"], db)
    print(f"[OK] {'Mise à jour' if overwrite else 'Ajout'} de '{args.name}' "
          f"(role={args.role}, dim={feat.shape[0]}, base={len(db)} personnes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
