"""
bytemark/ui/prof_watcher.py
QApplication-wide event filter that watches QDialog show/hide events
and toggles Prof.'s presence accordingly.

The watcher uses three independent safety nets so the Professor is
*guaranteed* to return to his workshop, no matter how a dialog ends:

  1. QEvent.Hide  / QEvent.Close   — normal teardown (user clicks X, accept,
                                      reject, or close via code).
  2. destroyed signal              — fires when Qt is destroying the QObject
                                      (parent reaped, garbage collected, etc).
  3. Idempotent counter on release — even if both fire, the presence manager
                                      ignores the duplicate release.

Install once at startup via `install_prof_watcher(app)`.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QApplication, QDialog

from bytemark.ui.prof_annotate import presence


class _ProfDialogWatcher(QObject):
    """Counts QDialog show/hide pairs so Prof. only returns to the
    workshop once every popup has closed."""

    def __init__(self, parent: QObject) -> None:
        super().__init__(parent)
        # Track dialogs we've summoned for. The set holds the id() of the
        # dialog QObject (cheap, no strong reference held).
        self._summoned: set[int] = set()

    # ── Event filter ─────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:  # noqa: D401
        if isinstance(obj, QDialog):
            t = event.type()
            key = id(obj)

            if t == QEvent.Type.Show and key not in self._summoned:
                self._summoned.add(key)
                presence().summon()
                # Safety net: even if Hide/Close never fire (parent reaped,
                # widget destroyed mid-flight, etc), the destroyed signal
                # still releases the presence.
                try:
                    obj.destroyed.connect(lambda *_, k=key: self._safe_release(k))
                except RuntimeError:
                    # `obj` already destroyed mid-event — release immediately.
                    self._safe_release(key)
            elif t in (QEvent.Type.Hide, QEvent.Type.Close):
                self._safe_release(key)
        return False

    # ── Internal release helper ──────────────────────────────────────────────

    def _safe_release(self, key: int) -> None:
        """Release presence for `key` exactly once. Safe to call any number of
        times from any of the three triggers."""
        if key in self._summoned:
            self._summoned.discard(key)
            presence().release()


def install_prof_watcher(app: QApplication) -> _ProfDialogWatcher:
    """Install the dialog watcher on the given QApplication. Returns the
    watcher so the caller can hold a reference (Qt won't keep it alive
    otherwise)."""
    watcher = _ProfDialogWatcher(app)
    app.installEventFilter(watcher)
    return watcher
