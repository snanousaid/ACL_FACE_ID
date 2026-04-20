"""Track vidéo WebRTC alimenté par le CameraWorker.

Le track lit la dernière frame annotée produite par le worker caméra et
la pousse sur la connexion WebRTC. Il n'ouvre PAS la caméra lui-même :
le worker est l'unique propriétaire du device.
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import av
import numpy as np
from aiortc import VideoStreamTrack

if TYPE_CHECKING:
    from camera_worker import CameraWorker


class FaceTrack(VideoStreamTrack):
    """VideoStreamTrack qui diffuse les frames du CameraWorker (BGR numpy)."""

    def __init__(self, worker: "CameraWorker", fps: int = 15) -> None:
        super().__init__()
        self._worker = worker
        self._frame_interval = 1.0 / max(1, int(fps))

    async def recv(self) -> av.VideoFrame:
        pts, time_base = await self.next_timestamp()

        frame = self._worker.get_frame_bgr()
        if frame is None:
            # Pas encore de frame produite par le worker → image noire temporaire
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Tempo : on ne dépasse pas le fps cible (aiortc ne throttle pas seul)
        await asyncio.sleep(self._frame_interval)

        video = av.VideoFrame.from_ndarray(frame, format="bgr24")
        video.pts = pts
        video.time_base = time_base
        return video
