"""
profannotate/ui/widgets/kpt_group_selector.py

Reusable keypoint-selection widget that scales to large schemas (e.g. the
133-kpt whole-body model). Renders collapsible region sections (Body, Face,
Feet, Left hand, Right hand …) plus one-click preset buttons derived from the
active schema's `groups`, on top of Select All / Deselect All.

`selected_names()` always returns names in canonical (schema) order.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from profannotate.config.skeleton import get_active_schema


class _CollapsibleSection(QWidget):
    """A header toggle that shows/hides a body of checkboxes."""

    def __init__(self, title: str, body: QWidget, start_open: bool) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(start_open)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if start_open else Qt.ArrowType.RightArrow)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.toggled.connect(self._on_toggled)

        self._body = body
        self._body.setVisible(start_open)

        layout.addWidget(self._toggle)
        layout.addWidget(self._body)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._body.setVisible(checked)


class KeypointGroupSelector(QWidget):
    """Grouped, preset-driven keypoint picker built from the active schema."""

    selectionChanged = Signal()  # emitted whenever any checkbox toggles

    def __init__(self, preselected: list[str] | None = None, scroll_height: int = 300) -> None:
        super().__init__()
        self._schema = get_active_schema()
        preselected_set = set(preselected) if preselected else None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Preset buttons — one per schema group (Face, Hands, Whole body, …).
        # Clicking a preset selects EXACTLY that group's keypoints.
        if self._schema.groups:
            presets_scroll = QScrollArea()
            presets_scroll.setFixedHeight(36)
            presets_scroll.setFrameShape(QFrame.Shape.NoFrame)
            presets_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            presets_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            presets_scroll.setWidgetResizable(True)

            presets_container = QWidget()
            presets_row = QHBoxLayout(presets_container)
            presets_row.setContentsMargins(0, 0, 0, 0)
            presets_row.setSpacing(6)
            presets_row.addWidget(QLabel("Presets:"))
            for label, members in self._schema.groups.items():
                btn = QPushButton(label)
                btn.setObjectName("preset_button")
                btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
                btn.clicked.connect(lambda _=False, m=list(members): self.set_selected(m))
                presets_row.addWidget(btn)
            presets_row.addStretch(1)
            presets_scroll.setWidget(presets_container)
            root.addWidget(presets_scroll)

        # Scrollable region sections.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(scroll_height)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        col = QVBoxLayout(container)
        col.setSpacing(6)

        self._checks: dict[str, QCheckBox] = {}
        sections = self._build_sections()
        for title, names in sections:
            body = QWidget()
            grid = QGridLayout(body)
            grid.setContentsMargins(14, 2, 2, 2)
            grid.setSpacing(2)
            # Pack many checkboxes into columns so big sections stay compact.
            cols = 3 if len(names) > 12 else 1
            for i, name in enumerate(names):
                cb = QCheckBox(name)
                cb.setChecked(preselected_set is None or name in preselected_set)
                cb.toggled.connect(self.selectionChanged)
                self._checks[name] = cb
                grid.addWidget(cb, i // cols, i % cols)
            # Big sections (face/hands) start collapsed to keep the dialog short.
            col.addWidget(_CollapsibleSection(f"{title}  ({len(names)})", body, len(names) <= 17))
        col.addStretch(1)
        scroll.setWidget(container)
        root.addWidget(scroll)

        # Select / Deselect all.
        sel_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_all.clicked.connect(lambda: self._set_all(True))
        desel_all = QPushButton("Deselect All")
        desel_all.clicked.connect(lambda: self._set_all(False))
        sel_row.addWidget(sel_all)
        sel_row.addWidget(desel_all)
        root.addLayout(sel_row)

    def _build_sections(self) -> list[tuple[str, list[str]]]:
        """Assign each keypoint to exactly one section (first group that owns it).

        Composite groups (e.g. 'Hands', 'Whole body') add no new members and so
        produce no section — they remain available only as preset buttons.
        """
        sections: list[tuple[str, list[str]]] = []
        assigned: set[str] = set()
        for label, members in self._schema.groups.items():
            fresh = [m for m in members if m not in assigned]
            if not fresh:
                continue
            sections.append((label, fresh))
            assigned.update(fresh)
        # Any keypoint not covered by a group falls into "Other".
        leftover = [n for n in self._schema.names_in_order() if n not in assigned]
        if leftover:
            sections.append(("Other", leftover))
        return sections

    def _set_all(self, checked: bool) -> None:
        for cb in self._checks.values():
            cb.setChecked(checked)

    def set_selected(self, names: list[str]) -> None:
        wanted = set(names)
        for name, cb in self._checks.items():
            cb.setChecked(name in wanted)

    def selected_names(self) -> list[str]:
        """Checked names in canonical schema order."""
        return [
            n
            for n in self._schema.names_in_order()
            if self._checks.get(n, None) and self._checks[n].isChecked()
        ]

    def has_any(self) -> bool:
        return any(cb.isChecked() for cb in self._checks.values())
