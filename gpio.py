"""Abstraction GPIO cross-platform.

- Windows (dev) : stub qui log, ne fait rien de physique.
- Linux A133 : pilote libgpiod v2 (via le paquet `gpiod` PyPI >=2.0) si installé,
  sinon fallback sur libgpiod v1, sinon stub.

Usage:
    from gpio import AccessActuator
    act = AccessActuator(chip="/dev/gpiochip1", line=7, active_high=True)
    act.pulse(3.0)   # non-bloquant
"""
from __future__ import annotations

import platform
import threading
import time
from typing import Optional


class AccessActuator:
    def __init__(
        self,
        chip: str = "/dev/gpiochip1",
        line: int = 7,
        active_high: bool = True,
        enabled: bool = True,
    ):
        self.chip_path = chip
        self.line_offset = int(line)
        self.active_high = bool(active_high)
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._backend = "stub"
        self._handle = None    # gpiod request (v2) ou line (v1)
        self._gpiod = None

        if not self.enabled or platform.system() != "Linux":
            return

        try:
            import gpiod
            self._gpiod = gpiod
        except ImportError:
            print("[GPIO] paquet 'gpiod' introuvable — mode stub (Windows/dev OK).")
            return

        # Tentative API v2 (gpiod >= 2.0)
        if hasattr(gpiod, "request_lines"):
            try:
                ls = gpiod.LineSettings(
                    direction=gpiod.line.Direction.OUTPUT,
                    output_value=gpiod.line.Value.INACTIVE,
                )
                self._handle = gpiod.request_lines(
                    self.chip_path,
                    consumer="access_ctrl",
                    config={self.line_offset: ls},
                )
                self._backend = "gpiod_v2"
                return
            except Exception as e:
                print(f"[GPIO] v2 init échoué ({e}) — essai v1")

        # Fallback API v1 (libgpiod 1.x)
        if hasattr(gpiod, "Chip"):
            try:
                chip_obj = gpiod.Chip(self.chip_path)
                ln = chip_obj.get_line(self.line_offset)
                ln.request(
                    consumer="access_ctrl",
                    type=getattr(gpiod, "LINE_REQ_DIR_OUT", 1),
                )
                self._handle = ln
                self._backend = "gpiod_v1"
                return
            except Exception as e:
                print(f"[GPIO] v1 init échoué ({e}) — mode stub")

    def describe(self) -> str:
        return (
            f"AccessActuator(backend={self._backend}, "
            f"chip={self.chip_path}, line={self.line_offset}, "
            f"active_high={self.active_high}, enabled={self.enabled})"
        )

    def pulse(self, seconds: float) -> None:
        """Active la gâche pendant `seconds` puis la coupe — non-bloquant.
        No-op complet si `enabled=False` (pas de stub print)."""
        if not self.enabled:
            return
        threading.Thread(target=self._pulse_impl, args=(float(seconds),), daemon=True).start()

    def _set(self, on: bool) -> None:
        if self._backend == "gpiod_v2":
            g = self._gpiod
            val = g.line.Value.ACTIVE if (on == self.active_high) else g.line.Value.INACTIVE
            self._handle.set_value(self.line_offset, val)
        elif self._backend == "gpiod_v1":
            val = 1 if (on == self.active_high) else 0
            self._handle.set_value(val)
        else:
            state = "HIGH" if on else "LOW"
            print(f"[GPIO STUB] line={self.line_offset} -> {state}")

    def _pulse_impl(self, seconds: float) -> None:
        with self._lock:
            self._set(True)
            try:
                time.sleep(seconds)
            finally:
                self._set(False)

    def close(self) -> None:
        try:
            if self._backend == "gpiod_v2" and self._handle is not None:
                self._handle.release()
            elif self._backend == "gpiod_v1" and self._handle is not None:
                self._handle.release()
        except Exception:
            pass
