"""
profannotate/ui/widgets/canvas.py
"""

from __future__ import annotations

import copy
from typing import Optional

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from profannotate.config import shortcuts as SC
from profannotate.config.constants import (
    ARROW_KEY_NUDGE_PX,
    CANVAS_SCROLL_SENSITIVITY,
    CANVAS_ZOOM_MAX,
    CANVAS_ZOOM_MIN,
    NUM_KEYPOINTS,
)
from profannotate.config.skeleton import KEYPOINT_NAMES
from profannotate.core.annotation.models import (
    Annotation,
    BBox,
    ImageAnnotations,
    Keypoint,
    Modality,
    SegmentationMask,
)
from profannotate.core.annotation.undo import UndoStack
from profannotate.ui.drawing.bbox_tool import BBoxTool
from profannotate.ui.drawing.keypoint_tool import KeypointTool
from profannotate.ui.drawing.segmentation_tool import SegmentationTool
from profannotate.ui.overlays.bbox_overlay import (
    HANDLE_BC,
    HANDLE_BL,
    HANDLE_BR,
    HANDLE_ML,
    HANDLE_MOVE,
    HANDLE_MR,
    HANDLE_NONE,
    HANDLE_TC,
    HANDLE_TL,
    HANDLE_TR,
    BBoxOverlay,
)
from profannotate.ui.overlays.diff_overlay import DiffOverlay
from profannotate.ui.overlays.keypoint_overlay import KeypointOverlay
from profannotate.ui.overlays.segmentation_overlay import SegmentationOverlay
from profannotate.utils.image import load_image_rgb, numpy_to_qpixmap


def _to_scene(view: QGraphicsView, event: QMouseEvent) -> QPointF:
    return view.mapToScene(event.position().toPoint())


class _LoaderSignals(QObject):
    """Cross-thread signal sink for image-load tasks. RGB array stays as numpy
    so the worker thread never touches QPixmap (which must live on the GUI
    thread). Generation is checked by the slot to drop stale results."""

    done = Signal(int, str, object)  # generation, path, rgb_array_or_None


class _ImageLoadTask(QRunnable):
    def __init__(self, generation: int, path: str, signals: _LoaderSignals) -> None:
        super().__init__()
        self._gen = generation
        self._path = path
        self._signals = signals
        self.setAutoDelete(True)

    def run(self) -> None:  # noqa: D401
        rgb = load_image_rgb(self._path)
        self._signals.done.emit(self._gen, self._path, rgb)


class _RgbCache:
    """Ordered-dict LRU keyed by image path, holding decoded RGB numpy arrays."""

    def __init__(self, capacity: int) -> None:
        from collections import OrderedDict

        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._capacity = capacity

    def get(self, key: str):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, val) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = val
            return
        self._cache[key] = val
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def clear(self) -> None:
        self._cache.clear()


class AnnotationCanvas(QFrame):
    annotations_changed = Signal(object)
    save_requested = Signal(object)
    undo_requested = Signal()
    undo_performed = Signal()
    image_loaded = Signal(str, int, int, bool, bool)
    auto_annotate_triggered = Signal()
    navigate_next = Signal()
    navigate_prev = Signal()
    instance_selected = Signal(object, int)
    instance_deselected = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("canvas_frame")

        self._annotations: Optional[ImageAnnotations] = None
        self._img_w = 1
        self._img_h = 1
        self._zoom = 1.0
        self._undo = UndoStack()
        self._dirty = False
        self._visible_modalities: set[Modality] = {
            Modality.BBOX,
            Modality.KEYPOINTS,
            Modality.SEGMENTATION,
        }

        self._num_keypoints: int = NUM_KEYPOINTS
        self._active_kpt_names: list[str] | None = None

        self._selected_instance: Optional[int] = None
        self._selected_kpt_idx: Optional[int] = None
        self._selected_pt_idx: Optional[int] = None

        self._dragging = False
        self._drag_type: Optional[str] = None
        self._drag_undo_pushed = False

        self._bbox_drag_handle: int = HANDLE_NONE
        self._bbox_drag_orig: Optional[BBox] = None
        self._bbox_kpts_orig: Optional[list] = None
        self._bbox_seg_orig: Optional[SegmentationMask] = None
        self._bbox_drag_scene_orig: Optional[QPointF] = None

        self._violation_active = False

        self._panning = False
        self._pan_origin: Optional[QPointF] = None

        self._bbox_tool: Optional[BBoxTool] = None
        self._kpt_tool: Optional[KeypointTool] = None
        self._seg_tool: Optional[SegmentationTool] = None
        self._active_draw_mode: Optional[str] = None

        self._bbox_items: list[BBoxOverlay] = []
        self._kpt_items: list[KeypointOverlay] = []
        self._seg_items: list[SegmentationOverlay] = []
        self._bbox_inst_map: list[int] = []
        self._kpt_inst_map: list[int] = []
        self._seg_inst_map: list[int] = []

        self._diff_item: Optional[DiffOverlay] = None
        self._seg_preview_item: Optional[SegmentationOverlay] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav = QWidget()
        nav.setFixedHeight(28)
        nav_layout = QGridLayout(nav)
        nav_layout.setContentsMargins(8, 0, 8, 0)
        nav_layout.setColumnStretch(0, 1)
        nav_layout.setColumnStretch(1, 0)
        nav_layout.setColumnStretch(2, 1)
        self._nav_label = QLabel("< 0/0 >")
        self._nav_label.setObjectName("nav_counter")
        self._draw_mode_label = QLabel("")
        self._draw_mode_label.setObjectName("draw_mode_indicator")
        self._draw_mode_label.hide()
        nav_layout.addWidget(self._nav_label, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        nav_layout.addWidget(
            self._draw_mode_label,
            0,
            2,
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
        )
        layout.addWidget(nav)

        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(self._view.renderHints().Antialiasing, True)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._view.setFrameShape(QFrame.Shape.NoFrame)
        self._view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self._view, stretch=1)

        self._view.viewport().installEventFilter(self)
        self._view.installEventFilter(self)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._set_border("unannotated")

    # ── Public API ────────────────────────────────────────────────────────────

    def load_entry(
        self,
        entry,
        num_keypoints: int = NUM_KEYPOINTS,
        active_kpt_names: list[str] | None = None,
    ) -> None:
        from profannotate.core.annotation.parser import parse_label_file

        self._num_keypoints = num_keypoints
        self._active_kpt_names = active_kpt_names
        ann = parse_label_file(entry.image_path, entry.label_path, num_keypoints)
        self._start_image_load(str(entry.image_path), ann)

    def set_annotations(self, ann: ImageAnnotations) -> None:
        self._annotations = ann
        self._rebuild_overlays()
        self._update_border()
        self.annotations_changed.emit(ann)

    def set_visible_modalities(self, modalities: set[Modality]) -> None:
        self._visible_modalities = modalities
        self._apply_selection_visibility()

    def set_nav_label(self, current: int, total: int) -> None:
        self._nav_label.setText(f"< {current}/{total} >")

    def show_diff(self, old: list[Annotation], new: list[Annotation]) -> None:
        self._clear_diff()
        self._diff_item = DiffOverlay(
            old, new, self._img_w, self._img_h, active_kpt_names=self._active_kpt_names
        )
        self._diff_item.setZValue(10)  # always above bbox/seg/kpt overlays
        self._scene.addItem(self._diff_item)
        # Dim the existing overlays so the diff reads clearly on top.
        for item in self._bbox_items + self._kpt_items + self._seg_items:
            item.setOpacity(0.25)
        self._draw_mode_label.setText("[ DIFF PREVIEW ]  Enter = accept  ·  Esc = reject")
        self._draw_mode_label.setStyleSheet("color: #FFD700;")
        self._draw_mode_label.show()
        # Reclaim focus — after _set_loading(False) re-enables the canvas,
        # tab-order focus lands on the first top-bar button instead of the
        # view, so Enter would activate it instead of accepting the diff.
        self._view.setFocus(Qt.FocusReason.OtherFocusReason)

    def accept_diff(self) -> None:
        if self._diff_item is not None:
            self._undo.push(self._annotations)
            self._annotations.instances = list(self._diff_item._new)
            self._clear_diff()
            self._rebuild_overlays()
            # Deselect first so instance_deselected fires before we re-select
            self._deselect()
            self._mark_dirty()
            # Select first instance and force JSON panel update
            if self._annotations.instances:
                self._select_instance(0)
                self.instance_selected.emit(self._annotations, 0)

    def reject_diff(self) -> None:
        self._clear_diff()
        self._draw_mode_label.setStyleSheet("")
        if self._active_draw_mode == "kpts":
            self._update_kpt_mode_label()
        elif self._active_draw_mode:
            self._draw_mode_label.setText(f"[ {self._active_draw_mode.upper()} MODE ]")
        else:
            self._draw_mode_label.hide()

    # ── Image loading ─────────────────────────────────────────────────────────

    # ── Pixmap cache + thread pool (lazy init) ────────────────────────────────

    def _ensure_loader(self) -> None:
        if getattr(self, "_rgb_cache", None) is not None:
            return
        # Cache capacity adapts to the active screen so low-end devices
        # don't carry 60 decoded RGB arrays in RAM (≈ hundreds of MB on
        # 4K images). Mid-tier laptops use 40, desktops 60.
        from profannotate.utils.ui_scaling import form_factor

        ff = form_factor()
        if ff == "tiny":
            cache_cap = 20
        elif ff == "small":
            cache_cap = 30
        elif ff == "medium":
            cache_cap = 40
        else:
            cache_cap = 60
        self._rgb_cache = _RgbCache(capacity=cache_cap)
        self._load_signals = _LoaderSignals()
        self._load_signals.done.connect(self._on_load_done, Qt.ConnectionType.QueuedConnection)
        self._load_gen = 0  # incremented on every foreground request
        self._foreground_path: Optional[str] = None
        self._pool = QThreadPool(self)
        # Two slots — one for the foreground request, one for opportunistic prefetch.
        # The OS file cache plus the LRU above handle the rest.
        self._pool.setMaxThreadCount(2)
        self._inflight: set[str] = set()

    def _start_image_load(self, path: str, ann: ImageAnnotations) -> None:
        self._ensure_loader()
        self._pending_ann = ann
        self._load_gen += 1
        gen = self._load_gen
        self._foreground_path = path

        cached = self._rgb_cache.get(path)
        if cached is not None:
            # Synchronous paint — bypass the worker entirely.
            self._render_rgb(gen, path, cached)
            return

        if path not in self._inflight:
            self._inflight.add(path)
            self._pool.start(_ImageLoadTask(gen, path, self._load_signals))

    def prefetch_paths(self, paths: list[str]) -> None:
        """Kick off background loads for `paths` that aren't already cached
        or already queued. Safe to call rapidly — duplicates are ignored.

        Skipped when the window isn't visible — no point burning disk I/O
        and decoder cycles for images the annotator isn't looking at."""
        window = self.window()
        if window is not None and not window.isVisible():
            return
        self._ensure_loader()
        for path in paths:
            if not path or path in self._rgb_cache or path in self._inflight:
                continue
            self._inflight.add(path)
            # generation 0 = prefetch (never matches the foreground gen, so
            # the slot only stores into the cache and does not repaint).
            self._pool.start(_ImageLoadTask(0, path, self._load_signals))

    def _on_load_done(self, gen: int, path: str, rgb) -> None:
        self._inflight.discard(path)
        if rgb is None:
            # Foreground load failed — surface the failure border.
            if gen == self._load_gen and path == self._foreground_path:
                self._on_image_failed()
            return
        self._rgb_cache.put(path, rgb)
        # Only render if this completion still matches the latest foreground request.
        if gen == self._load_gen and path == self._foreground_path:
            self._render_rgb(gen, path, rgb)

    def _render_rgb(self, gen: int, path: str, rgb) -> None:
        if gen != self._load_gen or path != self._foreground_path:
            return
        h, w = rgb.shape[:2]
        pixmap = numpy_to_qpixmap(rgb)
        self._on_image_loaded(pixmap, w, h)

    def _on_image_loaded(self, pixmap, w: int, h: int) -> None:
        self._img_w = w
        self._img_h = h
        self._bbox_items.clear()
        self._kpt_items.clear()
        self._seg_items.clear()
        self._bbox_inst_map.clear()
        self._kpt_inst_map.clear()
        self._seg_inst_map.clear()
        self._diff_item = None
        self._seg_preview_item = None
        self._scene.clear()

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(0, 0, w, h))
        self._view.fitInView(QRectF(0, 0, w, h), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = 1.0

        self._annotations = self._pending_ann
        self._selected_instance = None
        self._selected_kpt_idx = None
        self._selected_pt_idx = None
        self._dragging = False
        self._drag_type = None
        self._init_tools()
        self._rebuild_overlays()
        self._update_border()
        self._dirty = False
        self._undo.clear()
        self.instance_deselected.emit()

        ann = self._annotations
        self.image_loaded.emit(ann.image_path, w, h, ann.is_corrupted, ann.is_annotated())

    def _on_image_failed(self) -> None:
        self._set_border("unannotated")

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _init_tools(self) -> None:
        self._bbox_tool = BBoxTool(self._scene, self._img_w, self._img_h, self._on_bbox_drawn)
        self._kpt_tool = KeypointTool(
            self._scene,
            self._img_w,
            self._img_h,
            self._on_keypoint_placed,
            active_kpt_names=self._active_kpt_names,
        )
        self._seg_tool = SegmentationTool(
            self._scene, self._img_w, self._img_h, self._on_seg_closed
        )

    # ── Overlays ──────────────────────────────────────────────────────────────

    def _rebuild_overlays(self) -> None:
        scene = self._scene
        # Avoid the temporary list allocation from `a + b + c`.
        for item in self._bbox_items:
            scene.removeItem(item)
        for item in self._kpt_items:
            scene.removeItem(item)
        for item in self._seg_items:
            scene.removeItem(item)
        self._bbox_items.clear()
        self._kpt_items.clear()
        self._seg_items.clear()
        self._bbox_inst_map.clear()
        self._kpt_inst_map.clear()
        self._seg_inst_map.clear()

        if not self._annotations:
            return

        sel = self._selected_instance
        iw, ih = self._img_w, self._img_h
        akn = self._active_kpt_names
        for i, inst in enumerate(self._annotations.instances):
            if inst.has_bbox():
                item = BBoxOverlay(inst.bbox, iw, ih, inst.class_id, i)
                item.setZValue(1)
                item.set_selected(i == sel)
                scene.addItem(item)
                self._bbox_items.append(item)
                self._bbox_inst_map.append(i)
            if inst.has_keypoints():
                item = KeypointOverlay(
                    inst.keypoints,
                    iw,
                    ih,
                    i,
                    active_kpt_names=akn,
                )
                item.setZValue(2)
                scene.addItem(item)
                self._kpt_items.append(item)
                self._kpt_inst_map.append(i)
            if inst.has_mask():
                item = SegmentationOverlay(inst.mask, iw, ih, i)
                item.setZValue(1)
                scene.addItem(item)
                self._seg_items.append(item)
                self._seg_inst_map.append(i)

        self._apply_selection_visibility()
        if getattr(self, "_violation_active", False):
            violations = self._validate_annotations()
            self._mark_violations(violations)

    def _apply_selection_visibility(self) -> None:
        sel = self._selected_instance
        # Hoist the modality-membership checks out of the loops — previously
        # recomputed once per overlay item (3*N set lookups per call).
        vis = self._visible_modalities
        bbox_on = Modality.BBOX in vis
        kpt_on = Modality.KEYPOINTS in vis
        seg_on = Modality.SEGMENTATION in vis
        if sel is None:
            for item in self._bbox_items:
                item.setVisible(bbox_on)
            for item in self._kpt_items:
                item.setVisible(kpt_on)
            for item in self._seg_items:
                item.setVisible(seg_on)
        else:
            bb_map = self._bbox_inst_map
            kp_map = self._kpt_inst_map
            sg_map = self._seg_inst_map
            for j, item in enumerate(self._bbox_items):
                item.setVisible(bbox_on and bb_map[j] == sel)
            for j, item in enumerate(self._kpt_items):
                item.setVisible(kpt_on and kp_map[j] == sel)
            for j, item in enumerate(self._seg_items):
                item.setVisible(seg_on and sg_map[j] == sel)

    def _refresh_instance_overlays(self, inst_idx: int) -> None:
        if self._annotations is None:
            return
        inst = self._annotations.instances[inst_idx]
        for j, item in enumerate(self._bbox_items):
            if self._bbox_inst_map[j] == inst_idx and inst.bbox:
                item.update_bbox(inst.bbox)
        for j, item in enumerate(self._kpt_items):
            if self._kpt_inst_map[j] == inst_idx and inst.keypoints:
                item.update_keypoints(inst.keypoints)
                if self._drag_type == "kpt":
                    item.select_keypoint(self._selected_kpt_idx)
        for j, item in enumerate(self._seg_items):
            if self._seg_inst_map[j] == inst_idx and inst.mask:
                item.update_mask(inst.mask)
                if self._drag_type == "seg":
                    item.select_point(self._selected_pt_idx)

    def _clear_diff(self) -> None:
        if self._diff_item is not None:
            self._scene.removeItem(self._diff_item)
            self._diff_item = None
            for item in self._bbox_items + self._kpt_items + self._seg_items:
                item.setOpacity(1.0)

    def _update_kpt_mode_label(self) -> None:
        if self._active_draw_mode == "kpts" and self._kpt_tool:
            self._draw_mode_label.setText("[ KPTS MODE ]")
            self._refresh_kpt_cursor()

    def _refresh_kpt_cursor(self) -> None:
        if self._active_draw_mode != "kpts" or self._kpt_tool is None:
            return
        from profannotate.ui.drawing.keypoint_tool import _build_dot_cursor

        label = f"{self._kpt_tool._current_idx:02d} · {self._kpt_tool.current_name()}"
        viewport = self._view.viewport()
        viewport.unsetCursor()
        viewport.setCursor(_build_dot_cursor(label))

    def _create_seg_preview(self) -> None:
        if self._seg_preview_item is not None:
            self._scene.removeItem(self._seg_preview_item)
        self._seg_preview_item = SegmentationOverlay(
            SegmentationMask(), self._img_w, self._img_h, is_drawing=True
        )
        self._seg_preview_item.setZValue(3)
        self._scene.addItem(self._seg_preview_item)

    def _remove_seg_preview(self) -> None:
        if self._seg_preview_item is not None:
            self._scene.removeItem(self._seg_preview_item)
            self._seg_preview_item = None

    def _update_seg_preview(self) -> None:
        if self._seg_preview_item is None or self._seg_tool is None:
            return
        pts = self._seg_tool.in_progress_points
        mask = SegmentationMask(points=[(px / self._img_w, py / self._img_h) for px, py in pts])
        self._seg_preview_item.update_mask(mask)

    # ── Warning popups ────────────────────────────────────────────────────────

    def _show_warning(self, msg: str) -> None:
        """Surface a transient annotation error as a proper popup, Annotator.
        Replaces the older inline draw-mode flash."""
        from profannotate.ui.dialogs.error_dialog import ErrorDialog

        # Pick a sensible top-level parent so the popup centers on the window.
        ErrorDialog(msg, parent=self.window()).exec()

    # ── Border / dirty ────────────────────────────────────────────────────────

    def _set_border(self, state: str) -> None:
        self.setProperty("border_state", state)
        self.style().unpolish(self)
        self.style().polish(self)

    def _update_border(self) -> None:
        if self._annotations is None:
            self._set_border("unannotated")
        elif self._dirty:
            self._set_border("unsaved")
        elif self._annotations.is_annotated():
            self._set_border("annotated")
        else:
            self._set_border("unannotated")

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._set_border("unsaved")
        if self._annotations:
            self.annotations_changed.emit(self._annotations)

    def _validate_annotations(self) -> list[int]:
        if not self._annotations:
            return []
        violating: list[int] = []
        for i, inst in enumerate(self._annotations.instances):
            if not inst.has_bbox():
                continue
            x1, y1, x2, y2 = inst.bbox.to_xyxy(self._img_w, self._img_h)
            violated = False

            if inst.has_keypoints():
                for kp in inst.keypoints:
                    if kp is None or (kp.x == 0 and kp.y == 0 and kp.visibility == 0):
                        continue
                    px = kp.x * self._img_w
                    py = kp.y * self._img_h
                    if not (x1 <= px <= x2 and y1 <= py <= y2):
                        violated = True
                        break

            if not violated and inst.has_mask():
                for mx, my in inst.mask.points:
                    px = mx * self._img_w
                    py = my * self._img_h
                    if not (x1 <= px <= x2 and y1 <= py <= y2):
                        violated = True
                        break

            if violated:
                violating.append(i)
        return violating

    def _mark_violations(self, violating_indices: list[int]) -> None:
        indices_set = set(violating_indices)
        for j, item in enumerate(self._bbox_items):
            item.set_violated(self._bbox_inst_map[j] in indices_set)

    def _clear_violations(self) -> None:
        """Wipe the violation state — clears the red bbox highlights and
        resets the draw-mode label to its normal indicator."""
        self._violation_active = False
        for item in self._bbox_items:
            item.set_violated(False)
        self._draw_mode_label.setStyleSheet("")
        if self._active_draw_mode == "kpts":
            self._update_kpt_mode_label()
        elif self._active_draw_mode:
            self._draw_mode_label.setText(f"[ {self._active_draw_mode.upper()} MODE ]")
        else:
            self._draw_mode_label.hide()

    def _show_persistent_warning(self, msg: str) -> None:
        """Flag a violation that persists until corrected — the offending
        bboxes glow red and the Professor explains the issue in a popup."""
        from profannotate.ui.dialogs.error_dialog import ErrorDialog

        self._violation_active = True
        ErrorDialog(msg, parent=self.window()).exec()

    def _save(self) -> None:
        if self._annotations is None:
            return

        violations = self._validate_annotations()
        if violations:
            self._mark_violations(violations)
            count = len(violations)
            noun = "One entity has" if count == 1 else f"{count} entities have"
            self._show_persistent_warning(
                f"{noun} annotation points outside their bounding box, Annotator. "
                "Correct the violations before we can commit this to disk."
            )
            return

        self._clear_violations()
        from profannotate.core.annotation.writer import write_label_file

        write_label_file(self._annotations)
        self._dirty = False
        self._undo.clear()
        self._update_border()
        self.save_requested.emit(self._annotations)

    # ── Drawing callbacks ─────────────────────────────────────────────────────

    def _on_bbox_drawn(self, bbox: BBox, class_id: int) -> None:
        if self._annotations is None:
            return
        self._undo.push(self._annotations)
        self._annotations.add_instance(Annotation(class_id=class_id, bbox=bbox))
        new_idx = len(self._annotations.instances) - 1
        self._rebuild_overlays()
        self._select_instance(new_idx)
        self.instance_selected.emit(self._annotations, new_idx)
        self._mark_dirty()

    def _on_keypoint_placed(self, kpt_idx: int, kp: Keypoint) -> None:
        if self._annotations is None or self._selected_instance is None:
            return
        inst = self._annotations.instances[self._selected_instance]
        if inst.keypoints is None:
            inst.keypoints = [None] * self._num_keypoints
        # Pad up if the instance came from a file with fewer kpts than the dataset default.
        while len(inst.keypoints) <= kpt_idx:
            inst.keypoints.append(None)
        self._undo.push(self._annotations)
        inst.keypoints[kpt_idx] = kp
        self._rebuild_overlays()
        self._select_instance(self._selected_instance)
        self.instance_selected.emit(self._annotations, self._selected_instance)
        self._mark_dirty()
        self._update_kpt_mode_label()

    def _on_seg_closed(self, mask: SegmentationMask, class_id: int) -> None:
        if self._annotations is None or self._selected_instance is None:
            return
        inst = self._annotations.instances[self._selected_instance]
        self._undo.push(self._annotations)
        inst.mask = mask
        self._rebuild_overlays()
        self._select_instance(self._selected_instance)
        self.instance_selected.emit(self._annotations, self._selected_instance)
        self._mark_dirty()
        self._create_seg_preview()

    # ── Draw mode ─────────────────────────────────────────────────────────────

    def _activate_draw_mode(self, mode: str) -> None:
        if mode in ("kpts", "seg"):
            if (
                self._selected_instance is None
                or self._annotations is None
                or not self._annotations.instances[self._selected_instance].has_bbox()
            ):
                self._show_warning("A bounding box must be drawn and selected first, Annotator.")
                return

        self._deactivate_all_tools()
        self._active_draw_mode = mode
        self._draw_mode_label.setStyleSheet("")
        self._draw_mode_label.show()

        if mode == "bbox" and self._bbox_tool:
            self._bbox_tool.activate()
            self._draw_mode_label.setText("[ BBOX MODE ]")
        elif mode == "kpts" and self._kpt_tool:
            self._kpt_tool.activate()
            self._update_kpt_mode_label()
        elif mode == "seg" and self._seg_tool:
            self._seg_tool.activate()
            self._draw_mode_label.setText("[ SEG MODE ]")
            self._create_seg_preview()

    def _deactivate_all_tools(self) -> None:
        if self._bbox_tool:
            self._bbox_tool.deactivate()
        if self._kpt_tool:
            self._kpt_tool.deactivate()
        if self._seg_tool:
            self._seg_tool.deactivate()
        self._remove_seg_preview()
        self._active_draw_mode = None
        self._draw_mode_label.setStyleSheet("")
        self._draw_mode_label.hide()
        self._view.viewport().unsetCursor()

    def _toggle_draw_mode(self, mode: str) -> None:
        if self._active_draw_mode == mode:
            self._deactivate_all_tools()
        else:
            self._activate_draw_mode(mode)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _select_instance(self, inst_idx: int) -> None:
        for item in self._kpt_items:
            item.select_keypoint(None)
        for item in self._seg_items:
            item.select_point(None)
        for j, item in enumerate(self._bbox_items):
            item.set_selected(self._bbox_inst_map[j] == inst_idx)
        self._selected_instance = inst_idx
        self._apply_selection_visibility()

    def _deselect(self) -> None:
        self._selected_instance = None
        self._selected_kpt_idx = None
        self._selected_pt_idx = None
        self._dragging = False
        self._drag_type = None
        for item in self._bbox_items:
            item.set_selected(False)
        for item in self._kpt_items:
            item.select_keypoint(None)
        for item in self._seg_items:
            item.select_point(None)
        self._apply_selection_visibility()
        self.instance_deselected.emit()

    # ── Bbox-bounds check ─────────────────────────────────────────────────────

    def _point_inside_selected_bbox(self, scene_pos: QPointF) -> bool:
        if self._selected_instance is None or self._annotations is None:
            return True
        inst = self._annotations.instances[self._selected_instance]
        if not inst.has_bbox():
            return True
        x1, y1, x2, y2 = inst.bbox.to_xyxy(self._img_w, self._img_h)
        eps = 1.0
        return (x1 - eps) <= scene_pos.x() <= (x2 + eps) and (y1 - eps) <= scene_pos.y() <= (
            y2 + eps
        )

    # ── Undo ──────────────────────────────────────────────────────────────────

    def _undo_action(self) -> None:
        state = self._undo.undo()
        if state is not None:
            self._annotations = state
            self._selected_instance = None
            self._selected_kpt_idx = None
            self._selected_pt_idx = None
            self._dragging = False
            self._drag_type = None
            self._drag_undo_pushed = False
            self._bbox_drag_handle = HANDLE_NONE
            self._bbox_drag_orig = None
            self._bbox_kpts_orig = None
            self._bbox_seg_orig = None
            self._bbox_drag_scene_orig = None
            self._violation_active = False

            self._rebuild_overlays()
            self._mark_dirty()
            self.instance_deselected.emit()

            if self._active_draw_mode == "kpts" and self._kpt_tool and self._annotations.instances:
                inst = self._annotations.instances[0]
                if inst.keypoints:
                    next_idx = next(
                        (
                            i
                            for i, kp in enumerate(inst.keypoints)
                            if kp is None or (kp.x == 0 and kp.y == 0)
                        ),
                        0,
                    )
                    self._kpt_tool._current_idx = next_idx
                    self._update_kpt_mode_label()

        if self._active_draw_mode == "seg":
            self._create_seg_preview()

    # ── Event handling ────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        is_viewport = obj is self._view.viewport()
        is_view = obj is self._view
        if is_viewport or is_view:
            t = event.type()
            if t == QEvent.Type.Wheel and is_viewport:
                self._handle_wheel(event)
                return True
            if t == QEvent.Type.KeyPress:
                if self._handle_key(event):
                    return True
            if is_viewport:
                if t == QEvent.Type.MouseButtonDblClick:
                    if self._handle_double_click(event):
                        return True
                if t == QEvent.Type.MouseButtonPress:
                    self._handle_mouse_press(event)
                if t == QEvent.Type.MouseMove:
                    self._handle_mouse_move(event)
                if t == QEvent.Type.MouseButtonRelease:
                    self._handle_mouse_release(event)
        return super().eventFilter(obj, event)

    def _handle_wheel(self, event) -> None:
        delta = event.angleDelta().y()
        factor = CANVAS_SCROLL_SENSITIVITY if delta > 0 else 1.0 / CANVAS_SCROLL_SENSITIVITY
        nz = self._zoom * factor
        if CANVAS_ZOOM_MIN <= nz <= CANVAS_ZOOM_MAX:
            self._zoom = nz
            self._view.scale(factor, factor)

    def _handle_key(self, event: QKeyEvent) -> bool:
        key = event.key()
        mods = event.modifiers()

        if mods == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
            if key == Qt.Key.Key_A:
                self.auto_annotate_triggered.emit()
                return True

        if mods == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_S:
                self._save()
                return True
            if key == Qt.Key.Key_Z:
                if (
                    self._active_draw_mode == "seg"
                    and self._seg_tool
                    and self._seg_tool.point_count > 0
                ):
                    self._seg_tool.undo_last_point()
                    self._update_seg_preview()
                    return True
                self.undo_requested.emit()
                return True
            if key == Qt.Key.Key_Y:
                self.auto_annotate_triggered.emit()
                return True

        if key == Qt.Key.Key_Return and self._diff_item is not None:
            self.accept_diff()
            return True

        if key == Qt.Key.Key_Escape:
            if self._diff_item is not None:
                self.reject_diff()
            elif self._active_draw_mode:
                self._deactivate_all_tools()
            else:
                self._deselect()
            self._view.setFocus()
            return True

        if not mods:
            if key == SC.DRAW_BBOX:
                self._toggle_draw_mode("bbox")
                return True
            if key == SC.DRAW_KEYPOINT:
                self._toggle_draw_mode("kpts")
                return True
            if key == SC.DRAW_SEGMENTATION:
                self._toggle_draw_mode("seg")
                return True
            if key == Qt.Key.Key_D:
                self.navigate_next.emit()
                return True
            if key == Qt.Key.Key_A:
                self.navigate_prev.emit()
                return True

        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right):
            if self._selected_kpt_idx is not None or self._selected_pt_idx is not None:
                return self._nudge(key)
            if key == Qt.Key.Key_Left:
                self.navigate_prev.emit()
                return True
            if key == Qt.Key.Key_Right:
                self.navigate_next.emit()
                return True

        if key == Qt.Key.Key_Delete:
            return self._delete_selected()

        return False

    def _handle_double_click(self, event: QMouseEvent) -> bool:
        if event.button() != Qt.MouseButton.LeftButton:
            return False
        if self._active_draw_mode == "seg" and self._seg_tool:
            scene_pos = _to_scene(self._view, event)
            if self._seg_tool.double_click(scene_pos):
                self._remove_seg_preview()
                return True
        return False

    def _handle_mouse_press(self, event: QMouseEvent) -> None:
        # Right-click in kpt mode skips the current keypoint
        if event.button() == Qt.MouseButton.RightButton:
            if self._active_draw_mode == "kpts" and self._kpt_tool:
                self._kpt_tool.skip()
                self._update_kpt_mode_label()
            return

        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_origin = event.position()
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        scene_pos = _to_scene(self._view, event)

        if self._active_draw_mode == "bbox" and self._bbox_tool:
            self._bbox_tool.mouse_press(scene_pos)
        elif self._active_draw_mode == "kpts" and self._kpt_tool:
            if not self._point_inside_selected_bbox(scene_pos):
                self._show_warning("Keypoints must lie within the bounding box, Annotator.")
                return
            self._kpt_tool.mouse_press(scene_pos)
        elif self._active_draw_mode == "seg" and self._seg_tool:
            if not self._point_inside_selected_bbox(scene_pos):
                self._show_warning(
                    "Segmentation points must lie within the bounding box, Annotator."
                )
                return
            self._seg_tool.mouse_press(scene_pos)
            self._update_seg_preview()
        else:
            self._try_select_and_start_drag(scene_pos)

    def _handle_mouse_move(self, event: QMouseEvent) -> None:
        if self._panning and self._pan_origin is not None:
            delta = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self._view.horizontalScrollBar().setValue(
                self._view.horizontalScrollBar().value() - int(delta.x())
            )
            self._view.verticalScrollBar().setValue(
                self._view.verticalScrollBar().value() - int(delta.y())
            )
            return

        scene_pos = _to_scene(self._view, event)

        if self._active_draw_mode == "bbox" and self._bbox_tool:
            self._bbox_tool.mouse_move(scene_pos)
            return
        if self._active_draw_mode == "kpts" and self._kpt_tool:
            self._kpt_tool.set_zoom(self._zoom)
            self._kpt_tool.mouse_move(scene_pos)
            return
        if self._active_draw_mode == "seg" and self._seg_tool:
            self._seg_tool.mouse_move(scene_pos)
            return

        if self._dragging and self._annotations:
            self._drag_move(scene_pos)

    def _handle_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_origin = None
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        scene_pos = _to_scene(self._view, event)

        if self._active_draw_mode == "bbox" and self._bbox_tool:
            self._bbox_tool.mouse_release(scene_pos)
        elif self._active_draw_mode == "kpts" and self._kpt_tool:
            self._kpt_tool.mouse_release(scene_pos)
        elif self._active_draw_mode == "seg" and self._seg_tool:
            self._seg_tool.mouse_release(scene_pos)
        elif self._dragging:
            self._dragging = False
            self._drag_type = None
            self._drag_undo_pushed = False
            self._bbox_drag_handle = HANDLE_NONE
            if self._annotations:
                self._mark_dirty()
                if self._selected_instance is not None:
                    self.instance_selected.emit(self._annotations, self._selected_instance)

    # ── Select + drag ─────────────────────────────────────────────────────────

    def _try_select_and_start_drag(self, scene_pos: QPointF) -> None:
        if self._annotations is None:
            return

        for j, item in enumerate(self._kpt_items):
            if not item.isVisible():
                continue
            kpt_idx = item.hit_test_keypoint(scene_pos, self._zoom)
            if kpt_idx is not None:
                inst_idx = self._kpt_inst_map[j]
                self._select_instance(inst_idx)
                self._selected_kpt_idx = kpt_idx
                self._selected_pt_idx = None
                item.select_keypoint(kpt_idx)
                self._dragging = True
                self._drag_type = "kpt"
                self._drag_undo_pushed = False
                self.instance_selected.emit(self._annotations, inst_idx)
                return

        for j, item in enumerate(self._seg_items):
            if not item.isVisible():
                continue
            pt_idx = item.hit_test_point(scene_pos, self._zoom)
            if pt_idx is not None:
                inst_idx = self._seg_inst_map[j]
                self._select_instance(inst_idx)
                self._selected_pt_idx = pt_idx
                self._selected_kpt_idx = None
                item.select_point(pt_idx)
                self._dragging = True
                self._drag_type = "seg"
                self._drag_undo_pushed = False
                self.instance_selected.emit(self._annotations, inst_idx)
                return

        if self._selected_instance is not None:
            for j, item in enumerate(self._bbox_items):
                if self._bbox_inst_map[j] == self._selected_instance:
                    handle = item.hit_test_handle(scene_pos)
                    if handle not in (HANDLE_NONE, HANDLE_MOVE):
                        self._start_bbox_drag(handle, scene_pos)
                        return
                    break

        if self._selected_instance is not None:
            for j, item in enumerate(self._bbox_items):
                if self._bbox_inst_map[j] == self._selected_instance:
                    if item.hit_test_handle(scene_pos) == HANDLE_MOVE:
                        self._start_bbox_drag(HANDLE_MOVE, scene_pos)
                        return
                    break

        for j, item in enumerate(self._bbox_items):
            if not item.isVisible():
                continue
            if item._rect.contains(scene_pos):
                inst_idx = self._bbox_inst_map[j]
                self._select_instance(inst_idx)
                self._selected_kpt_idx = None
                self._selected_pt_idx = None
                self.instance_selected.emit(self._annotations, inst_idx)
                return

        self._deselect()

    def _start_bbox_drag(self, handle: int, scene_pos: QPointF) -> None:
        if self._annotations is None or self._selected_instance is None:
            return
        inst = self._annotations.instances[self._selected_instance]
        self._drag_type = "bbox"
        self._bbox_drag_handle = handle
        self._bbox_drag_orig = copy.deepcopy(inst.bbox)
        self._bbox_kpts_orig = copy.deepcopy(inst.keypoints) if inst.keypoints else None
        self._bbox_seg_orig = copy.deepcopy(inst.mask) if inst.mask else None
        self._bbox_drag_scene_orig = scene_pos
        self._dragging = True
        self._drag_undo_pushed = False

    def _drag_move(self, scene_pos: QPointF) -> None:
        if self._annotations is None or self._selected_instance is None:
            return
        inst = self._annotations.instances[self._selected_instance]

        if self._drag_type == "kpt" and self._selected_kpt_idx is not None:
            if not (inst.keypoints and self._selected_kpt_idx < len(inst.keypoints)):
                return
            if not self._drag_undo_pushed:
                self._undo.push(self._annotations)
                self._drag_undo_pushed = True
            kp = inst.keypoints[self._selected_kpt_idx]
            vis = kp.visibility if kp is not None else 2
            clamped = self._clamp_to_instance_bbox(scene_pos, self._selected_instance)
            inst.keypoints[self._selected_kpt_idx] = Keypoint.from_pixel(
                clamped.x(), clamped.y(), self._img_w, self._img_h, vis
            )
            self._refresh_instance_overlays(self._selected_instance)

        elif self._drag_type == "seg" and self._selected_pt_idx is not None:
            if not (inst.mask and self._selected_pt_idx < len(inst.mask.points)):
                return
            if not self._drag_undo_pushed:
                self._undo.push(self._annotations)
                self._drag_undo_pushed = True
            clamped = self._clamp_to_instance_bbox(scene_pos, self._selected_instance)
            nx = max(0.0, min(1.0, clamped.x() / self._img_w))
            ny = max(0.0, min(1.0, clamped.y() / self._img_h))
            inst.mask.update_point(self._selected_pt_idx, nx, ny)
            self._refresh_instance_overlays(self._selected_instance)

        elif self._drag_type == "bbox" and self._bbox_drag_orig is not None:
            if not self._drag_undo_pushed:
                self._undo.push(self._annotations)
                self._drag_undo_pushed = True
            self._drag_move_bbox(scene_pos, inst)

    def _clamp_to_instance_bbox(self, scene_pos: QPointF, inst_idx: int) -> QPointF:
        if self._annotations is None:
            return scene_pos
        inst = self._annotations.instances[inst_idx]
        if not inst.has_bbox():
            return scene_pos
        x1, y1, x2, y2 = inst.bbox.to_xyxy(self._img_w, self._img_h)
        return QPointF(
            max(x1, min(x2, scene_pos.x())),
            max(y1, min(y2, scene_pos.y())),
        )

    def _drag_move_bbox(self, scene_pos: QPointF, inst: Annotation) -> None:
        orig = self._bbox_drag_orig
        ox1, oy1, ox2, oy2 = orig.to_xyxy(self._img_w, self._img_h)
        tdx = scene_pos.x() - self._bbox_drag_scene_orig.x()
        tdy = scene_pos.y() - self._bbox_drag_scene_orig.y()
        nx1, ny1, nx2, ny2 = ox1, oy1, ox2, oy2
        h = self._bbox_drag_handle
        W, H = float(self._img_w), float(self._img_h)

        if h == HANDLE_MOVE:
            nx1 = ox1 + tdx
            ny1 = oy1 + tdy
            nx2 = ox2 + tdx
            ny2 = oy2 + tdy
            if nx1 < 0:
                shift = -nx1
                nx1 += shift
                nx2 += shift
            if ny1 < 0:
                shift = -ny1
                ny1 += shift
                ny2 += shift
            if nx2 > W:
                shift = nx2 - W
                nx1 -= shift
                nx2 -= shift
            if ny2 > H:
                shift = ny2 - H
                ny1 -= shift
                ny2 -= shift
        else:
            min_size = 4.0
            if h in (HANDLE_TL, HANDLE_ML, HANDLE_BL):
                nx1 = min(ox2 - min_size, ox1 + tdx)
            if h in (HANDLE_TR, HANDLE_MR, HANDLE_BR):
                nx2 = max(ox1 + min_size, ox2 + tdx)
            if h in (HANDLE_TL, HANDLE_TC, HANDLE_TR):
                ny1 = min(oy2 - min_size, oy1 + tdy)
            if h in (HANDLE_BL, HANDLE_BC, HANDLE_BR):
                ny2 = max(oy1 + min_size, oy2 + tdy)
            nx1 = max(0.0, nx1)
            ny1 = max(0.0, ny1)
            nx2 = min(W, nx2)
            ny2 = min(H, ny2)

        inst.bbox = BBox.from_xyxy(nx1, ny1, nx2, ny2, self._img_w, self._img_h)
        self._refresh_instance_overlays(self._selected_instance)

    # ── Nudge / delete ────────────────────────────────────────────────────────

    def _nudge(self, key: int) -> bool:
        if self._annotations is None or self._selected_instance is None:
            return False
        inst = self._annotations.instances[self._selected_instance]

        if self._selected_kpt_idx is not None:
            if not (inst.keypoints and self._selected_kpt_idx < len(inst.keypoints)):
                return False
            kp = inst.keypoints[self._selected_kpt_idx]
            if kp is None:
                return False
            dx = dy = 0
            n = ARROW_KEY_NUDGE_PX
            if key == Qt.Key.Key_Up:
                dy = -n
            if key == Qt.Key.Key_Down:
                dy = n
            if key == Qt.Key.Key_Left:
                dx = -n
            if key == Qt.Key.Key_Right:
                dx = n
            self._undo.push(self._annotations)
            clamped = self._clamp_to_instance_bbox(
                QPointF(kp.x * self._img_w + dx, kp.y * self._img_h + dy),
                self._selected_instance,
            )
            inst.keypoints[self._selected_kpt_idx] = Keypoint.from_pixel(
                clamped.x(), clamped.y(), self._img_w, self._img_h, kp.visibility
            )
            self._refresh_instance_overlays(self._selected_instance)
            self._mark_dirty()
            return True

        if self._selected_pt_idx is not None:
            if not (inst.mask and self._selected_pt_idx < len(inst.mask.points)):
                return False
            x, y = inst.mask.points[self._selected_pt_idx]
            dx = dy = 0.0
            n = ARROW_KEY_NUDGE_PX / self._img_w
            if key == Qt.Key.Key_Up:
                dy = -n
            if key == Qt.Key.Key_Down:
                dy = n
            if key == Qt.Key.Key_Left:
                dx = -n
            if key == Qt.Key.Key_Right:
                dx = n
            self._undo.push(self._annotations)
            clamped = self._clamp_to_instance_bbox(
                QPointF((x + dx) * self._img_w, (y + dy) * self._img_h),
                self._selected_instance,
            )
            inst.mask.update_point(
                self._selected_pt_idx,
                max(0.0, min(1.0, clamped.x() / self._img_w)),
                max(0.0, min(1.0, clamped.y() / self._img_h)),
            )
            self._refresh_instance_overlays(self._selected_instance)
            self._mark_dirty()
            return True

        return False

    def _delete_selected(self) -> bool:
        if self._annotations is None or self._selected_instance is None:
            return False
        inst = self._annotations.instances[self._selected_instance]

        if self._selected_kpt_idx is not None:
            if inst.keypoints and self._selected_kpt_idx < len(inst.keypoints):
                self._undo.push(self._annotations)
                inst.keypoints[self._selected_kpt_idx] = Keypoint(0, 0, 0)
                self._refresh_instance_overlays(self._selected_instance)
                self._selected_kpt_idx = None
                self._mark_dirty()
                return True

        if self._selected_pt_idx is not None:
            if inst.mask:
                self._undo.push(self._annotations)
                inst.mask.remove_point(self._selected_pt_idx)
                self._refresh_instance_overlays(self._selected_instance)
                self._selected_pt_idx = None
                self._mark_dirty()
                return True

        from profannotate.ui.dialogs.confirm_dialog import ConfirmDialog

        parts = []
        if inst.has_bbox():
            parts.append("bounding box")
        if inst.has_keypoints():
            parts.append("keypoints")
        if inst.has_mask():
            parts.append("segmentation mask")
        body = (
            "Annotator, you are about to permanently remove this annotation instance, "
            "which contains:\n• "
            + "\n• ".join(parts)
            + "\n\nThis action will be added to the undo stack, but proceed with care."
        )
        dlg = ConfirmDialog("Remove Annotation Instance?", body, "> Yes, remove it", "No, keep it")
        if dlg.exec() != dlg.DialogCode.Accepted:
            return False
        self._undo.push(self._annotations)
        self._annotations.remove_instance(self._selected_instance)
        self._deselect()
        self._rebuild_overlays()
        self._mark_dirty()
        return True
