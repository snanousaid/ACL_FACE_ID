"""Bridge pour émettre les événements d'accès vers les clients WebSocket.

`CameraWorker` appelle `emit(...)` à chaque accès (granted/denied). `webapp.py`
enregistre un callback qui pousse l'événement sur le serveur socket.io:5001.
Si aucun émetteur n'est enregistré (mode CLI `main.py`), `emit` est no-op.
"""
from __future__ import annotations

from typing import Callable, Optional

_emitter: Optional[Callable[[dict], None]] = None


def set_emitter(fn: Callable[[dict], None]) -> None:
    global _emitter
    _emitter = fn


def emit(payload: dict) -> None:
    if _emitter is None:
        return
    try:
        _emitter(payload)
    except Exception:
        pass
