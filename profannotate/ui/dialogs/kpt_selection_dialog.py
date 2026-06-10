"""
profannotate/ui/dialogs/kpt_selection_dialog.py
Standalone modal that asks which keypoints should be active for a dataset.
Used when opening an images-only dataset that has no recorded kpt config yet.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from profannotate.config.constants import NUM_KEYPOINTS
from profannotate.config.skeleton import KEYPOINT_NAMES


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

        chosen_w = screen_aware_size(frame, preferred_w=600, min_w=360, parent=parent)
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
            "Choose which keypoints should be available for drawing — the choice "
            "is written to data.yaml and used everywhere from auto-annotation to "
            "the skeleton overlay.\n\n"
            "All keypoints are selected by default."
        )
        body.setObjectName("dialog_body")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(body)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(280)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        kw = QWidget()
        kl = QVBoxLayout(kw)
        kl.setSpacing(3)

        preselected_set = set(preselected) if preselected else None
        self._checks: dict[int, QCheckBox] = {}
        for idx in range(NUM_KEYPOINTS):
            name = KEYPOINT_NAMES.get(idx, str(idx))
            cb = QCheckBox(f"  {idx:02d}  {name}")
            cb.setChecked(preselected_set is None or name in preselected_set)
            kl.addWidget(cb)
            self._checks[idx] = cb
        scroll.setWidget(kw)
        inner.addWidget(scroll)

        sel_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_all.clicked.connect(lambda: [cb.setChecked(True) for cb in self._checks.values()])
        desel_all = QPushButton("Deselect All")
        desel_all.clicked.connect(lambda: [cb.setChecked(False) for cb in self._checks.values()])
        sel_row.addWidget(sel_all)
        sel_row.addWidget(desel_all)
        inner.addLayout(sel_row)

        self._err = QLabel("")
        self._err.setObjectName("accent_red")
        self._err.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self._err)

        btn_row = QHBoxLayout()
        ok = QPushButton("> Save selection")
        ok.setObjectName("primary_button")
        ok.setDefault(True)
        ok.setAutoDefault(True)
        ok.clicked.connect(self._on_ok)
        cancel = QPushButton("Use all keypoints")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self._on_use_all)
        btn_row.addWidget(ok)
        btn_row.addWidget(cancel)
        inner.addLayout(btn_row)
        self._focus_target = ok

        outer.addWidget(frame, alignment=Qt.AlignmentFlag.AlignCenter)

    def showEvent(self, event) -> None:  # noqa: D401
        super().showEvent(event)
        self._focus_target.setFocus(Qt.FocusReason.OtherFocusReason)

    def keyPressEvent(self, event) -> None:  # noqa: D401
        if event.key() == Qt.Key.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def _on_ok(self) -> None:
        chosen = [KEYPOINT_NAMES[i] for i, cb in sorted(self._checks.items()) if cb.isChecked()]
        if not chosen:
            self._err.setText("At least one keypoint must be selected, Annotator.")
            return
        self._selected = chosen
        self.accept()

    def _on_use_all(self) -> None:
        self._selected = [KEYPOINT_NAMES[i] for i in sorted(KEYPOINT_NAMES)]
        self.accept()

    def selected_names(self) -> list[str] | None:
        return self._selected
