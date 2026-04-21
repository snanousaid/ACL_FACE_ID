"""Shared memory writer — envoie les frames traitées vers Qt sans copie réseau.

Layout du bloc mémoire partagée :
  [0..3]   frame_id  uint32  — incrément à chaque frame écrite
  [4..7]   width     uint32
  [8..11]  height    uint32
  [12]     channels  uint8   — toujours 3 (BGR)
  [13]     ready     uint8   — 0=écriture en cours, 1=frame prête
  [14..15] padding   2 bytes
  [16...]  frame     width * height * channels bytes (BGR contiguous)
"""
from __future__ import annotations

import struct
import logging
from multiprocessing.shared_memory import SharedMemory

import numpy as np

SHM_NAME   = "acl_video_stream"
HEADER_SIZE = 16  # octets de métadonnées avant les pixels

logger = logging.getLogger("shm_writer")


class ShmWriter:
    """Écrit des frames OpenCV dans une zone mémoire partagée.

    Usage:
        writer = ShmWriter(640, 480)
        writer.write(frame)   # frame numpy BGR HxWx3
        writer.close()
    """

    def __init__(self, width: int, height: int, channels: int = 3) -> None:
        self.width    = width
        self.height   = height
        self.channels = channels
        self._frame_id = 0

        frame_bytes = width * height * channels
        total_size  = HEADER_SIZE + frame_bytes

        try:
            # Tente de récupérer un bloc existant (redémarrage)
            self._shm = SharedMemory(name=SHM_NAME, create=False, size=total_size)
            logger.info("ShmWriter: bloc existant récupéré (%d bytes)", total_size)
        except FileNotFoundError:
            self._shm = SharedMemory(name=SHM_NAME, create=True, size=total_size)
            logger.info("ShmWriter: nouveau bloc créé (%d bytes)", total_size)

        # Vue numpy directe sur les pixels (zero-copy)
        self._pixels = np.ndarray(
            (height, width, channels),
            dtype=np.uint8,
            buffer=self._shm.buf,
            offset=HEADER_SIZE,
        )

    def write(self, frame: np.ndarray) -> None:
        """Écrit une frame dans la mémoire partagée."""
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = _resize(frame, self.width, self.height)

        # ready = 0 → écriture en cours
        self._shm.buf[13] = 0

        # Copie pixels (inévitable si frame non contiguë)
        np.copyto(self._pixels, frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame))

        self._frame_id += 1

        # Écriture header
        struct.pack_into("<IIIBBB", self._shm.buf, 0,
                         self._frame_id,   # frame_id
                         self.width,
                         self.height,
                         self.channels,
                         1,                # ready = 1
                         0)                # padding

    def close(self) -> None:
        self._shm.close()
        try:
            self._shm.unlink()
        except Exception:
            pass
        logger.info("ShmWriter: bloc libéré")


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


# Import tardif pour éviter les dépendances circulaires
import cv2  # noqa: E402
