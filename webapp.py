"""Interface web d'administration multi-utilisateurs avec stream vidéo live.

Lancer: python webapp.py
  → HTTP (dashboard/stream/API) : http://localhost:5000
  → Socket.IO (événements face) : ws://localhost:5001

ATTENTION: la webapp possède maintenant la caméra via CameraWorker.
Ne pas lancer main.py en parallèle (conflit caméra).
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import socketio
from flask import (
    Flask, Response, flash, jsonify, redirect, render_template, request, url_for,
)
from flask_cors import CORS
from werkzeug.serving import make_server

import face_events
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

# --------------- Socket.IO (port 5001) ----------------------
sio = socketio.Server(cors_allowed_origins="*", async_mode="threading")
sio_app = socketio.WSGIApp(sio)
_sio_log = logging.getLogger("face-socket")


@sio.event
def connect(sid, environ):  # noqa: D401
    _sio_log.info("client connecté: %s", sid)


@sio.event
def disconnect(sid):
    _sio_log.info("client déconnecté: %s", sid)


def _emit_face_event(payload: dict) -> None:
    """Callback branché dans face_events : pousse sur tous les clients."""
    sio.emit("event", payload)


face_events.set_emitter(_emit_face_event)

# --------------- Worker caméra -------------------------------
WORKER = CameraWorker(CFG)

# --------------- Flask (port 5000) ---------------------------
app = Flask(__name__)
# Autorise tous les domaines (ACL_Terminal dev = localhost:5173, prod = Electron file://).
CORS(app)
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
    s = WORKER.get_status()
    # Champs aplatis pour consommation React (ACL_Terminal)
    if s.get("enroll_poses"):
        s["poses_done"] = [p["id"] for p in s["enroll_poses"] if p.get("done")]
        s["required_poses"] = [p["id"] for p in s["enroll_poses"] if p.get("required")]
        s["optional_poses"] = [p["id"] for p in s["enroll_poses"] if not p.get("required")]
        s["current_pose"] = s.get("enroll_current_pose")
    return jsonify(s)


# --------------- API JSON utilisateurs (consommée par ACL_Terminal) ---------------
def _list_users() -> list[dict]:
    db = load_known(CFG["paths"]["embeddings"])
    out = []
    for name, entry in sorted(db.items()):
        out.append({
            "name": name,
            "role": entry.get("role", "user"),
            "created_at": entry.get("created_at", ""),
            "active": entry.get("active", True),
            "dim": int(np.asarray(entry["embedding"]).shape[0]),
        })
    return out


@app.route("/api/users", methods=["GET"])
def api_users():
    return jsonify(_list_users())


@app.route("/api/users/<name>/toggle", methods=["POST"])
def api_toggle_user(name: str):
    db = load_known(CFG["paths"]["embeddings"])
    if name not in db:
        return jsonify({"ok": False, "msg": "introuvable"}), 404
    db[name]["active"] = not db[name].get("active", True)
    save_known(CFG["paths"]["embeddings"], db)
    WORKER.reload_db()
    return jsonify({"ok": True, "active": db[name]["active"]})


@app.route("/api/users/<name>", methods=["DELETE"])
def api_delete_user(name: str):
    db = load_known(CFG["paths"]["embeddings"])
    if name not in db:
        return jsonify({"ok": False, "msg": "introuvable"}), 404
    del db[name]
    save_known(CFG["paths"]["embeddings"], db)
    WORKER.reload_db()
    return jsonify({"ok": True})


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


# --------------- gestion utilisateurs (formulaires HTML legacy) ---------------
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


# --------------- serveur Socket.IO (thread séparé) ---------------
def _start_socketio_server(host: str = "0.0.0.0", port: int = 5001) -> threading.Thread:
    server = make_server(host, port, sio_app, threaded=True)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="face-socketio")
    t.start()
    logging.info("Socket.IO face events → ws://%s:%s", host, port)
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")
    _start_socketio_server()
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        WORKER.stop()
