"""
profannotate/ui/widgets/kpt_selection_panel.py

Guided, two-step keypoint-selection flow shared by the standalone selection
dialog and the dataset wizard:

  Step 1 — declare how many keypoints (N) will be annotated.
  Step 2 — pick exactly N from the grouped list, with a live skeleton diagram
           showing which keypoints were chosen and where they sit on the body.

The panel owns its flow buttons and emits:
  proceeded(object)  → list[str] (the chosen subset) | None (use all keypoints)
  cancelled()        → user backed out
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from profannotate.config.skeleton import get_active_schema
from profannotate.ui.widgets.kpt_group_selector import KeypointGroupSelector
from profannotate.ui.widgets.skeleton_preview import SkeletonPreview


class KeypointSelectionPanel(QWidget):
    proceeded = Signal(object)  # list[str] | None
    cancelled = Signal()

    def __init__(self, preselected: list[str] | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._schema = get_active_schema()
        self._preselected = preselected
        self._N = len(preselected) if preselected else self._schema.count

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_count_step())
        self._stack.addWidget(self._build_select_step(preselected))

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._stack)

    # ── Step 1: count ───────────────────────────────────────────────────────────

    def _build_count_step(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(14)

        prompt = QLabel(
            "How many keypoints will you annotate in this dataset, Annotator?\n\n"
            f"This schema offers up to {self._schema.count} keypoints. Pick a target "
            "count now; on the next step you'll choose exactly that many."
        )
        prompt.setObjectName("dialog_body")
        prompt.setWordWrap(True)
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(prompt)

        spin_row = QHBoxLayout()
        spin_row.addStretch(1)
        spin_row.addWidget(QLabel("Number of keypoints:"))
        self._spin = QSpinBox()
        self._spin.setRange(1, self._schema.count)
        self._spin.setValue(self._N)
        self._spin.setFixedWidth(90)
        spin_row.addWidget(self._spin)
        spin_row.addStretch(1)
        v.addLayout(spin_row)

        v.addStretch(1)

        btns = QHBoxLayout()
        cont = QPushButton("> Continue")
        cont.setObjectName("primary_button")
        cont.setDefault(True)
        cont.clicked.connect(self._on_continue)
        use_all = QPushButton(f"Use all {self._schema.count} keypoints")
        use_all.clicked.connect(lambda: self.proceeded.emit(None))
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.cancelled)
        btns.addWidget(cont)
        btns.addWidget(use_all)
        btns.addWidget(cancel)
        v.addLayout(btns)
        return page

    # ── Step 2: choose ────────────────────────────────────────────────────────────

    def _build_select_step(self, preselected: list[str] | None) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setSpacing(10)

        body = QHBoxLayout()
        self._selector = KeypointGroupSelector(preselected=preselected, scroll_height=300)
        self._selector.selectionChanged.connect(self._on_selection_changed)
        body.addWidget(self._selector, 3)
        self._preview = SkeletonPreview()
        body.addWidget(self._preview, 2)
        v.addLayout(body)

        self._counter = QLabel("")
        self._counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._counter)

        btns = QHBoxLayout()
        back = QPushButton("< Back")
        back.clicked.connect(lambda: self._stack.setCurrentIndex(0))
        self._proceed = QPushButton("> Proceed with selected keypoints")
        self._proceed.setObjectName("primary_button")
        self._proceed.clicked.connect(self._on_proceed)
        use_all = QPushButton("Use all keypoints")
        use_all.clicked.connect(lambda: self.proceeded.emit(None))
        btns.addWidget(back)
        btns.addWidget(self._proceed)
        btns.addWidget(use_all)
        v.addLayout(btns)
        return page

    # ── Flow ──────────────────────────────────────────────────────────────────────

    def _on_continue(self) -> None:
        self._N = self._spin.value()
        # For a fresh subset (no explicit preselection), start the picker empty so
        # the annotator builds up to exactly N instead of unchecking the full set.
        if self._preselected is None and self._N < self._schema.count:
            self._selector.set_selected([])
        self._stack.setCurrentIndex(1)
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        sel = self._selector.selected_names()
        self._preview.set_selected(sel)
        matched = len(sel) == self._N
        self._counter.setText(f"Selected {len(sel)} / {self._N}")
        self._counter.setObjectName("" if matched else "accent_red")
        # re-polish so the objectName-based stylesheet applies
        self._counter.style().unpolish(self._counter)
        self._counter.style().polish(self._counter)
        self._proceed.setEnabled(matched)

    def _on_proceed(self) -> None:
        sel = self._selector.selected_names()
        if len(sel) != self._N:
            return
        # Selecting the full schema is equivalent to "use all".
        self.proceeded.emit(None if len(sel) == self._schema.count else sel)
