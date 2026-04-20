# ACL_FACE_ID — Contrôle d'accès par reconnaissance faciale

Système embarqué de contrôle d'accès par reconnaissance faciale, conçu pour tourner sur carte **Allwinner A733** (HelperA733, aarch64) avec dev/test sur **Windows x64**.

**Stack** : OpenCV DNN (YuNet + SFace ONNX) + Flask + libgpiod.

**Repo** : https://github.com/snanousaid/ACL_FACE_ID

---

## Architecture

```
┌─────────────┐     V4L2/DSHOW     ┌──────────────┐
│   Caméra    │ ──────────────────► │ CameraWorker │
│  USB/CSI    │                    │  (thread bg) │
└─────────────┘                    │              │
                                   │  YuNet ──► Détection visage
                                   │  SFace ──► Embedding 128-D
                                   │  cosine ──► Match vs known_faces.pkl
                                   │              │
                                   │  ┌── GRANTED ──► GPIO pulse (gâche)
                                   │  └── DENIED  ──► log
                                   │              │
                                   │  frame BGR ──► FaceTrack (aiortc VP8)
                                   └──────┬───────┘
                                          │
                         ┌────────────────┼─────────────────┐
                         │          Flask webapp             │
                         │  /               → dashboard      │
                         │  /webrtc/offer   → signalisation  │
                         │  /status.json    → statut temps réel
                         │  /enroll_*       → enrôlement     │
                         │  /pose_thumb/*   → vignettes poses│
                         └────────────────┼─────────────────┘
                                          │
                                    http://:5050
                                     navigateur
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `config.yaml` | Configuration centralisée (caméra, seuils, perf, GPIO, poses) |
| `utils.py` | Helpers : caméra cross-platform, FaceProcessor (YuNet+SFace), match cosinus, luminosité |
| `camera_worker.py` | Thread unique : capture → détection → reco → stream MJPEG → enrôlement multi-poses |
| `gpio.py` | Abstraction GPIO : stub silencieux Windows, libgpiod v2/v1 sur Linux |
| `webapp.py` | Flask : dashboard, stream vidéo, CRUD utilisateurs, enrôlement live/upload |
| `main.py` | Mode headless CLI (sans web, pour déploiement pur terminal) |
| `enroll.py` | Enrôlement CLI : `python enroll.py <nom> --role admin --samples 15` |
| `templates/index.html` | UI web : stream live, statut accès, gestion multi-users, poses, logs |
| `requirements.txt` | Dépendances avec markers cross-platform (numpy, opencv, Flask, gpiod) |

## Installation

### Windows (dev/test)

```bash
cd ACL_FACE_ID
pip install -r requirements.txt

# Télécharger les modèles ONNX (une seule fois) :
mkdir models
cd models
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
curl -LO https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
cd ..

python webapp.py
# → http://localhost:5050
```

### Linux A733 (production)

```bash
git clone https://github.com/snanousaid/ACL_FACE_ID.git
cd ACL_FACE_ID

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

mkdir -p models && cd models
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
cd ..

# Vérifier la caméra :
v4l2-ctl --list-devices
ls -l /dev/video*
# Si permission denied : sudo usermod -aG video $USER && logout

python3 webapp.py
# → http://<IP_board>:5050
```

## Utilisation

### Interface web (http://\<IP\>:5050)

- **Stream live** : flux MJPEG temps réel avec bbox et labels
- **Bandeau accès** : ACCÈS AUTORISÉ (vert) / REFUSÉ (rouge) / EN ATTENTE (gris)
- **Statut** : nom, rôle, score, luminosité — poll 400ms
- **Enrôlement via caméra** : 3 poses obligatoires (face, gauche, droite) + 1 optionnelle (bas). Vignettes par pose. Pas de sauvegarde si poses incomplètes (logique iPhone Face ID).
- **Enrôlement via upload** : drag-drop d'images, extraction automatique
- **Gestion** : activer/désactiver/supprimer des utilisateurs, rôles (admin/user/visitor)
- **Logs** : 25 dernières lignes du journal d'accès

### Mode CLI (headless)

```bash
# Enrôlement
python enroll.py said --role admin --samples 15

# Reconnaissance continue (sans web)
python main.py
```

## Configuration (config.yaml)

```yaml
camera:
  index: 0          # /dev/video0
  width: 640
  height: 480
  fps: 15

recognition:
  match_threshold: 0.70    # cosine similarity

perf:
  opencv_threads: 2        # limiter pour ARM
  detect_every_n: 2        # skip-frame
  detect_scale: 1.0        # 0.5 = détection sur demi-résolution (requiert OpenCV >=4.8)
  stream_fps: 10
  # (JPEG encoding retiré : stream passé en WebRTC/VP8)

enrollment:
  required_poses: ["center", "left", "right"]
  optional_poses: ["down"]

gpio:
  enabled: false           # true quand gâche câblée
  chip: "/dev/gpiochip1"
  line: 7
  active_high: true
```

## Compatibilité

| | Windows (dev) | Linux A733 (prod) |
|---|---|---|
| Python | 3.10 — 3.14 | 3.8 — 3.12 |
| Arch | x86_64 | aarch64 |
| Caméra | DSHOW (auto) | V4L2 (auto) |
| GPIO | stub silencieux | libgpiod v2/v1 |
| numpy | ≥2.1 (env marker) | <2.0 si Py<3.10 |

## Problèmes connus

| Problème | Cause | Solution |
|---|---|---|
| `detect_scale: 0.5` crash | OpenCV 4.6 DNN shape mismatch | Garder 1.0 ou upgrader OpenCV ≥4.8 |
| `known_faces.pkl` incompatible | numpy 1.x ↔ 2.x pickle mismatch | `rm embeddings/known_faces.pkl` + ré-enrôler |
| Pose "haut" pas détectée | Estimation pitch peu fiable en 2D | Retirée des poses par défaut |
| `Camera introuvable` | USB débranchée ou permissions | `sudo usermod -aG video $USER` |
| CPU élevé sur A733 | Détection chaque frame + 8 threads | `opencv_threads: 2` + `detect_every_n: 2` |

## Historique

| Commit | Description |
|---|---|
| `f430161` | Initial : YuNet + SFace + Flask + enrôlement multi-poses |
| `2cbb260` | Abstraction GPIO cross-platform |
| `35e5562` | Optimisation CPU A733 (~3× moins) |
| `b89a396` | Qualité stream : 640×480 capture + detect_scale + vignettes |
| `c322457` | Poses haut/bas optionnelles |
| `df5cbd0` | Gestion numpy pickle incompatible |
| `6b454ce` | Fallback caméra robuste (scan auto) |
