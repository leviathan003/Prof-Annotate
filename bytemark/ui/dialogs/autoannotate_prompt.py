"""
bytemark/ui/dialogs/autoannotate_prompt.py
Ctrl+Y single-image auto-annotation prompt.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from bytemark.core.annotation.models import Modality
from bytemark.ui.dialogs.modality_prompt import ModalityPrompt


class AutoAnnotatePrompt(ModalityPrompt):
    """Single-image auto-annotate dialog — reuses ModalityPrompt."""

    def __init__(self, parent=None) -> None:
        super().__init__(
            title="Auto-Annotate This Image",
            show_warning=True,
            parent=parent,
        )
