"""Interface web d'administration multi-utilisateurs avec stream vidéo live.

Lancer: python webapp.py  →  http://localhost:5000

ATTENTION: la webapp possède maintenant la caméra via CameraWorker.
Ne pas lancer main.py en parallèle (conflit caméra).
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, Response, flash, jsonify, redirect, render_template, request, url_for,
)

from camera_worker import CameraWorker
from utils import (
    FaceProcessor,
    ensure_dirs,
    load_config,
    load_known,
    save_known,
)

CFG = load_config("config.yaml")
ensure_dirs(CFG)
WORKER = CameraWorker(CFG)

app = Flask(__name__)
app.secret_key = "change-me-in-production"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# --------------- pages ---------------
@app.route("/")
def index():
    db = load_known(CFG["paths"]["embeddings"])
    users = []
    for name, entry in sorted(db.items()):
        users.append({
            "name": name,
            "role": entry.get("role", "user"),
            "created_at": entry.get("created_at", ""),
            "active": entry.get("active", True),
            "dim": int(np.asarray(entry["embedding"]).shape[0]),
        })

    log_path = Path(CFG["paths"]["log_file"])
    logs: list[str] = []
    if log_path.exists():
        logs = log_path.read_text(encoding="utf-8").splitlines()[-25:][::-1]

    return render_template(
        "index.html",
        users=users,
        logs=logs,
        threshold=CFG["recognition"]["match_threshold"],
    )


# --------------- stream vidéo MJPEG ---------------
_STREAM_INTERVAL = 1.0 / max(1, int(CFG.get("perf", {}).get("stream_fps", 10)))


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpg = WORKER.get_jpeg()
        if jpg is None:
            time.sleep(0.05)
            continue
        yield (boundary + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        time.sleep(_STREAM_INTERVAL)


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# --------------- statut temps réel ---------------
@app.route("/status.json")
def status_json():
    return jsonify(WORKER.get_status())


# --------------- enrôlement par upload (hors caméra) ---------------
def _extract_embeddings(files) -> tuple[list[np.ndarray], list[str]]:
    feats, warnings = [], []
    for f in files:
        if not f or not f.filename:
            continue
        data = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            warnings.append(f"{f.filename}: décodage impossible")
            continue
        faces = WORKER.fp.detect(img)
        if faces is None or len(faces) == 0:
            warnings.append(f"{f.filename}: aucun visage détecté")
            continue
        best = FaceProcessor.best_face(faces)
        feats.append(WORKER.fp.embed(img, best))
    return feats, warnings


@app.route("/enroll_upload", methods=["POST"])
def enroll_upload():
    name = (request.form.get("name") or "").strip()
    role = (request.form.get("role") or "user").strip()
    files = request.files.getlist("images")

    if not name or not files:
        flash("Nom et images requis.", "error")
        return redirect(url_for("index"))

    feats, warnings = _extract_embeddings(files)
    if not feats:
        flash("Aucun visage exploitable. " + " | ".join(warnings), "error")
        return redirect(url_for("index"))

    mean_feat = np.mean(np.stack(feats), axis=0).astype(np.float32)
    db = load_known(CFG["paths"]["embeddings"])
    existed = name in db
    db[name] = {
        "embedding": mean_feat, "role": role,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "active": True,
    }
    save_known(CFG["paths"]["embeddings"], db)
    WORKER.reload_db()
    flash(f"{'Mis à jour' if existed else 'Ajouté'}: {name} [{role}] — {len(feats)} échantillon(s).", "success")
    return redirect(url_for("index"))


# --------------- enrôlement LIVE (caméra) ---------------
@app.route("/enroll_live/start", methods=["POST"])
def enroll_live_start():
    name = (request.form.get("name") or "").strip()
    role = (request.form.get("role") or "user").strip()
    # N échantillons PAR POSE (5 poses → total = n_per_pose * 5)
    n_per_pose = int(request.form.get("samples_per_pose") or 5)
    if not name:
        return jsonify({"ok": False, "msg": "Nom requis."}), 400
    ok, msg = WORKER.start_enroll(n_per_pose, name, role)
    return jsonify({"ok": ok, "msg": msg})


@app.route("/enroll_live/finalize", methods=["POST"])
def enroll_live_finalize():
    ok, msg = WORKER.finalize_enroll()
    return jsonify({"ok": ok, "msg": msg})


@app.route("/enroll_live/cancel", methods=["POST"])
def enroll_live_cancel():
    WORKER.cancel_enroll()
    return jsonify({"ok": True, "msg": "Annulé."})


@app.route("/pose_thumb/<pose>.jpg")
def pose_thumb(pose: str):
    data = WORKER.get_pose_thumb(pose)
    if data is None:
        return ("", 404)
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# --------------- gestion utilisateurs ---------------
@app.route("/delete/<name>", methods=["POST"])
def delete(name: str):
    db = load_known(CFG["paths"]["embeddings"])
    if name in db:
        del db[name]
        save_known(CFG["paths"]["embeddings"], db)
        WORKER.reload_db()
        flash(f"'{name}' supprimé.", "success")
    return redirect(url_for("index"))


@app.route("/toggle/<name>", methods=["POST"])
def toggle(name: str):
    db = load_known(CFG["paths"]["embeddings"])
    if name in db:
        db[name]["active"] = not db[name].get("active", True)
        save_known(CFG["paths"]["embeddings"], db)
        WORKER.reload_db()
        flash(f"'{name}' {'activé' if db[name]['active'] else 'désactivé'}.", "success")
    return redirect(url_for("index"))


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        WORKER.stop()
