"""
bytemark/ui/widgets/modality_selector.py
Three toggle buttons controlling which annotation modalities are visible.
Emits modalities_changed(set[Modality]) on every toggle.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from src.core.annotation.models import Modality


class ModalitySelector(QWidget):
    modalities_changed = Signal(set)  # set[Modality]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._buttons: dict[Modality, QPushButton] = {}

        specs = [
            (Modality.BBOX, "BBox", "modality_bbox"),
            (Modality.KEYPOINTS, "Kpts", "modality_kpts"),
            (Modality.SEGMENTATION, "Mask", "modality_seg"),
        ]
        for modality, label, obj_name in specs:
            btn = QPushButton(f"• {label}")
            btn.setObjectName(obj_name)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggled.connect(self._on_toggle)
            btn.setToolTip(f"Toggle {label} visibility  Ctrl+{1 + list(self._buttons).__len__()}")
            layout.addWidget(btn)
            self._buttons[modality] = btn

        layout.addStretch()

    def _on_toggle(self) -> None:
        self.modalities_changed.emit(self.active_modalities())

    def active_modalities(self) -> set[Modality]:
        return {m for m, btn in self._buttons.items() if btn.isChecked()}

    def set_modality_visible(self, modality: Modality, visible: bool) -> None:
        if modality in self._buttons:
            self._buttons[modality].setChecked(visible)
