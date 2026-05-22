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

from src.core.annotation.models import Modality
from src.ui.dialogs.modality_prompt import ModalityPrompt


class AutoAnnotatePrompt(ModalityPrompt):
    def __init__(self, parent=None) -> None:
        super().__init__(
            title="Auto-Annotate This Image, Annotator.",
            show_warning=True,
            parent=parent,
        )
