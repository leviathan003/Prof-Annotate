"""
profannotate/ui/dialogs/kpt_selection_dialog.py
Standalone modal that asks which keypoints should be active for a dataset.
Used when opening an images-only dataset that has no recorded kpt config yet.

Wraps the shared two-step `KeypointSelectionPanel` (declare count → pick exactly
that many, with a live skeleton diagram). `selected_names()` returns the chosen
list in canonical order, or None if cancelled.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QLabel,
    QVBoxLayout,
)

from profannotate.ui.widgets.kpt_selection_panel import KeypointSelectionPanel


class KptSelectionDialog(QDialog):
    """Lets the user choose which keypoints to annotate.

    `selected_names()` returns the chosen list (always in canonical order),
    or None if the dialog was cancelled.
    """

    def __init__(self, parent=None, preselected: list[str] | None = None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._selected: list[str] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        frame = QFrame()
        frame.setObjectName("overlay_dialog")
        from profannotate.ui.dialogs._prof_layout import screen_aware_size

        chosen_w = screen_aware_size(frame, preferred_w=720, min_w=480, parent=parent)
        frame.setMinimumWidth(chosen_w)
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(28, 24, 28, 24)
        inner.setSpacing(14)

        title = QLabel("Select Keypoints to Annotate")
        title.setObjectName("dialog_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(title)

        body = QLabel(
            "This dataset has no recorded keypoint configuration yet, Annotator. "
            "Declare how many keypoints you'll annotate, then choose exactly those "
            "— the choice is written to data.yaml and used everywhere from "
            "auto-annotation to the skeleton overlay."
        )
        body.setObjectName("dialog_body")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(body)

        self._panel = KeypointSelectionPanel(preselected=preselected)
        self._panel.proceeded.connect(self._on_proceeded)
        self._panel.cancelled.connect(self.reject)
        inner.addWidget(self._panel)

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def _on_proceeded(self, names) -> None:
        # None = "use all keypoints" → expand to the full active schema.
        if names is None:
            from profannotate.config.skeleton import get_active_schema

            self._selected = get_active_schema().names_in_order()
        else:
            self._selected = list(names)
        self.accept()

    def selected_names(self) -> list[str] | None:
        return self._selected
