"""
bytemark/core/annotation/undo.py
10-step in-memory undo. Cleared on save. No file-level undo.

`collections.deque` with `maxlen` gives O(1) push + automatic FIFO drop —
the previous `list.pop(0)` was O(N) on every overflow. Trivial at N=10 but
free perf nonetheless.
"""

from __future__ import annotations

import copy
import threading
from collections import deque
from typing import Optional

from bytemark.config.constants import UNDO_HISTORY_SIZE
from bytemark.core.annotation.models import ImageAnnotations


class UndoStack:
    def __init__(self, max_size: int = UNDO_HISTORY_SIZE) -> None:
        self._stack: deque[ImageAnnotations] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def push(self, state: ImageAnnotations) -> None:
        with self._lock:
            self._stack.append(copy.deepcopy(state))

    def undo(self) -> Optional[ImageAnnotations]:
        with self._lock:
            return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        with self._lock:
            self._stack.clear()

    @property
    def can_undo(self) -> bool:
        return len(self._stack) > 0

    @property
    def depth(self) -> int:
        return len(self._stack)
