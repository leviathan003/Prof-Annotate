"""
bytemark/ui/widgets/canvas.py
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from bytemark.config import shortcuts as SC
from bytemark.config.constants import (
    ARROW_KEY_NUDGE_PX,
    CANVAS_SCROLL_SENSITIVITY,
    CANVAS_ZOOM_MAX,
    CANVAS_ZOOM_MIN,
)
from bytemark.core.annotation.models import (
    Annotation,
    BBox,
    ImageAnnotations,
    Keypoint,
    Modality,
    SegmentationMask,
)
from bytemark.core.annotation.undo import UndoStack
from bytemark.ui.drawing.bbox_tool import BBoxTool
from bytemark.ui.drawing.keypoint_tool import KeypointTool
from bytemark.ui.drawing.segmentation_tool import SegmentationTool
from bytemark.ui.overlays.bbox_overlay import BBoxOverlay
from bytemark.ui.overlays.diff_overlay import DiffOverlay
from bytemark.ui.overlays.keypoint_overlay import KeypointOverlay
from bytemark.ui.overlays.segmentation_overlay import SegmentationOverlay
from bytemark.utils.image import load_image_rgb, numpy_to_qpixmap


class _ImageLoader(QObject):
    loaded = Signal(object, int, int)
    failed = Signal()

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path

    def run(self) -> None:
        rgb = load_image_rgb(self._path)
        if rgb is None:
            self.failed.emit()
            return
        h, w = rgb.shape[:2]
        self.loaded.emit(numpy_to_qpixmap(rgb), w, h)


class AnnotationCanvas(QFrame):
    annotations_changed = Signal(object)
    save_requested = Signal(object)
    undo_performed = Signal()
    image_loaded = Signal(str, int, int, bool, bool)
    auto_annotate_triggered = Signal()
    navigate_next = Signal()
    navigate_prev = Signal()
    instance_selected = Signal(object, int)  # (ImageAnnotations, instance_idx)
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

        # Selection state
        self._selected_instance: Optional[int] = None
        self._selected_kpt_idx: Optional[int] = None
        self._selected_pt_idx: Optional[int] = None

        # Drag state
        self._dragging = False
        self._drag_type: Optional[str] = None  # 'kpt' | 'seg'
        self._drag_undo_pushed = False

        self._panning = False
        self._pan_origin = None

        self._bbox_tool: Optional[BBoxTool] = None
        self._kpt_tool: Optional[KeypointTool] = None
        self._seg_tool: Optional[SegmentationTool] = None
        self._active_draw_mode: Optional[str] = None

        # Overlay items — parallel to instances (one entry per qualifying instance)
        self._bbox_items: list[BBoxOverlay] = []
        self._kpt_items: list[KeypointOverlay] = []
        self._seg_items: list[SegmentationOverlay] = []
        # Maps item-list index → instance index
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
        nav_layout = QHBoxLayout(nav)
        nav_layout.setContentsMargins(8, 0, 8, 0)
        self._nav_label = QLabel("< 0/0 >")
        self._nav_label.setObjectName("nav_counter")
        self._draw_mode_label = QLabel("")
        self._draw_mode_label.setObjectName("draw_mode_indicator")
        self._draw_mode_label.hide()
        nav_layout.addWidget(self._nav_label)
        nav_layout.addStretch()
        nav_layout.addWidget(self._draw_mode_label)
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
        self._placeholder = self._scene.addText(
            "image viewer / annotation editor window\n\n"
            "Open a dataset folder to begin.\n\n"
            "• B — bbox  • K — keypoints  • S — segmentation\n"
            "• Ctrl+S — save  • Ctrl+Z — undo  • Ctrl+Y — auto-annotate\n"
            "• A / D or ← / → — previous / next image\n"
            "• Click annotation — isolate & drag points  • Esc — show all\n"
            "• Middle mouse — pan  • Scroll — zoom\n"
            "• S mode: single click = add point, double click = close mask"
        )
        self._placeholder.setDefaultTextColor(QColor("#2A2A2A"))
        self._set_border("unannotated")

    # ── Public API ────────────────────────────────────────────────────────────

    def load_entry(self, entry) -> None:
        from bytemark.core.annotation.parser import parse_label_file

        ann = parse_label_file(entry.image_path, entry.label_path)
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
        self._diff_item = DiffOverlay(old, new, self._img_w, self._img_h)
        self._scene.addItem(self._diff_item)

    def accept_diff(self) -> None:
        if self._diff_item is not None:
            self._undo.push(self._annotations)
            self._annotations.instances = self._diff_item._new
            self._clear_diff()
            self._rebuild_overlays()
            self._mark_dirty()

    def reject_diff(self) -> None:
        self._clear_diff()

    # ── Image loading ─────────────────────────────────────────────────────────

    def _start_image_load(self, path: str, ann: ImageAnnotations) -> None:
        self._pending_ann = ann
        self._thread = QThread()
        self._loader = _ImageLoader(path)
        self._loader.moveToThread(self._thread)
        self._thread.started.connect(self._loader.run)
        self._loader.loaded.connect(self._on_image_loaded, Qt.ConnectionType.QueuedConnection)
        self._loader.failed.connect(self._on_image_failed, Qt.ConnectionType.QueuedConnection)
        self._loader.loaded.connect(self._thread.quit)
        self._loader.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_image_loaded(self, pixmap: QPixmap, w: int, h: int) -> None:
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
            self._scene, self._img_w, self._img_h, self._on_keypoint_placed
        )
        self._seg_tool = SegmentationTool(
            self._scene, self._img_w, self._img_h, self._on_seg_closed
        )

    # ── Overlays ──────────────────────────────────────────────────────────────

    def _rebuild_overlays(self) -> None:
        for item in self._bbox_items + self._kpt_items + self._seg_items:
            self._scene.removeItem(item)
        self._bbox_items.clear()
        self._kpt_items.clear()
        self._seg_items.clear()
        self._bbox_inst_map.clear()
        self._kpt_inst_map.clear()
        self._seg_inst_map.clear()

        if not self._annotations:
            return

        for i, inst in enumerate(self._annotations.instances):
            if inst.has_bbox():
                item = BBoxOverlay(inst.bbox, self._img_w, self._img_h, inst.class_id, i)
                item.setZValue(1)
                self._scene.addItem(item)
                self._bbox_items.append(item)
                self._bbox_inst_map.append(i)
            if inst.has_keypoints():
                item = KeypointOverlay(inst.keypoints, self._img_w, self._img_h, i)
                item.setZValue(2)
                self._scene.addItem(item)
                self._kpt_items.append(item)
                self._kpt_inst_map.append(i)
            if inst.has_mask():
                item = SegmentationOverlay(inst.mask, self._img_w, self._img_h, i)
                item.setZValue(1)
                self._scene.addItem(item)
                self._seg_items.append(item)
                self._seg_inst_map.append(i)

        self._apply_selection_visibility()

    def _apply_selection_visibility(self) -> None:
        """Show all overlays normally, or isolate to selected instance."""
        if self._selected_instance is None:
            for j, item in enumerate(self._bbox_items):
                item.setVisible(Modality.BBOX in self._visible_modalities)
            for j, item in enumerate(self._kpt_items):
                item.setVisible(Modality.KEYPOINTS in self._visible_modalities)
            for j, item in enumerate(self._seg_items):
                item.setVisible(Modality.SEGMENTATION in self._visible_modalities)
        else:
            for j, item in enumerate(self._bbox_items):
                item.setVisible(
                    self._bbox_inst_map[j] == self._selected_instance
                    and Modality.BBOX in self._visible_modalities
                )
            for j, item in enumerate(self._kpt_items):
                item.setVisible(
                    self._kpt_inst_map[j] == self._selected_instance
                    and Modality.KEYPOINTS in self._visible_modalities
                )
            for j, item in enumerate(self._seg_items):
                item.setVisible(
                    self._seg_inst_map[j] == self._selected_instance
                    and Modality.SEGMENTATION in self._visible_modalities
                )

    def _clear_diff(self) -> None:
        if self._diff_item is not None:
            self._scene.removeItem(self._diff_item)
            self._diff_item = None

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

    def _save(self) -> None:
        if self._annotations is None:
            return
        from bytemark.core.annotation.writer import write_label_file

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
        self._rebuild_overlays()
        self._mark_dirty()

    def _on_keypoint_placed(self, kpt_idx: int, kp: Keypoint) -> None:
        if self._annotations is None:
            return
        if not self._annotations.instances:
            self._undo.push(self._annotations)
            self._annotations.add_instance(Annotation(class_id=0, keypoints=[None] * 19))
        inst = self._annotations.instances[-1]
        if inst.keypoints is None:
            inst.keypoints = [None] * 19
        self._undo.push(self._annotations)
        inst.keypoints[kpt_idx] = kp
        self._rebuild_overlays()
        self._mark_dirty()

    def _on_seg_closed(self, mask: SegmentationMask, class_id: int) -> None:
        if self._annotations is None:
            return
        self._undo.push(self._annotations)
        self._annotations.add_instance(Annotation(class_id=class_id, mask=mask))
        self._rebuild_overlays()
        self._mark_dirty()
        self._create_seg_preview()

    # ── Draw mode ─────────────────────────────────────────────────────────────

    def _activate_draw_mode(self, mode: str) -> None:
        self._deactivate_all_tools()
        self._active_draw_mode = mode
        self._draw_mode_label.setText(f"[ {mode.upper()} MODE ]")
        self._draw_mode_label.show()
        if mode == "bbox" and self._bbox_tool:
            self._bbox_tool.activate()
        elif mode == "kpts" and self._kpt_tool:
            self._kpt_tool.activate()
        elif mode == "seg" and self._seg_tool:
            self._seg_tool.activate()
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
        self._draw_mode_label.hide()

    def _toggle_draw_mode(self, mode: str) -> None:
        if self._active_draw_mode == mode:
            self._deactivate_all_tools()
        else:
            self._activate_draw_mode(mode)

    # ── Deselect (ESC when not in draw mode) ──────────────────────────────────

    def _deselect(self) -> None:
        self._selected_instance = None
        self._selected_kpt_idx = None
        self._selected_pt_idx = None
        self._dragging = False
        self._drag_type = None
        for item in self._kpt_items:
            item.select_keypoint(None)
        for item in self._seg_items:
            item.select_point(None)
        self._apply_selection_visibility()
        self.instance_deselected.emit()

    # ── Undo ──────────────────────────────────────────────────────────────────

    def _undo_action(self) -> None:
        state = self._undo.undo()
        if state is not None:
            self._annotations = state
            self._rebuild_overlays()
            self._mark_dirty()

    # ── Event handling ────────────────────────────────────────────────────────

    def eventFilter(self, obj, event) -> bool:
        if obj is self._view or obj is self._view.viewport():
            t = event.type()
            if t == QEvent.Type.Wheel:
                self._handle_wheel(event)
                return True
            if t == QEvent.Type.KeyPress:
                if self._handle_key(event):
                    return True
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

        if mods == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_S:
                self._save()
                return True
            if key == Qt.Key.Key_Z:
                self._undo_action()
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
            scene_pos = self._view.mapToScene(event.position().toPoint())
            if self._seg_tool.double_click(scene_pos):
                self._remove_seg_preview()
                return True
        return False

    def _handle_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_origin = event.position().toPoint()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        scene_pos = self._view.mapToScene(event.position().toPoint())

        if self._active_draw_mode == "bbox" and self._bbox_tool:
            self._bbox_tool.mouse_press(scene_pos)
        elif self._active_draw_mode == "kpts" and self._kpt_tool:
            self._kpt_tool.mouse_press(scene_pos)
        elif self._active_draw_mode == "seg" and self._seg_tool:
            self._seg_tool.mouse_press(scene_pos)
            self._update_seg_preview()
        else:
            self._try_select_and_start_drag(scene_pos)

    def _handle_mouse_move(self, event: QMouseEvent) -> None:
        if self._panning and self._pan_origin is not None:
            delta = event.position().toPoint() - self._pan_origin
            self._pan_origin = event.position().toPoint()
            self._view.horizontalScrollBar().setValue(
                self._view.horizontalScrollBar().value() - delta.x()
            )
            self._view.verticalScrollBar().setValue(
                self._view.verticalScrollBar().value() - delta.y()
            )
            return

        scene_pos = self._view.mapToScene(event.position().toPoint())

        if self._active_draw_mode == "bbox" and self._bbox_tool:
            self._bbox_tool.mouse_move(scene_pos)
            return
        if self._active_draw_mode == "kpts" and self._kpt_tool:
            self._kpt_tool.mouse_move(scene_pos)
            return
        if self._active_draw_mode == "seg" and self._seg_tool:
            self._seg_tool.mouse_move(scene_pos)
            return

        # Drag selected point
        if self._dragging and self._annotations:
            self._drag_move(scene_pos)

    def _handle_mouse_release(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_origin = None
            return

        scene_pos = self._view.mapToScene(event.position().toPoint())

        if event.button() == Qt.MouseButton.LeftButton:
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
                if self._annotations:
                    self._mark_dirty()
                    if self._selected_instance is not None:
                        self.instance_selected.emit(self._annotations, self._selected_instance)

    # ── Selection + drag ──────────────────────────────────────────────────────

    def _try_select_and_start_drag(self, scene_pos: QPointF) -> None:
        """Hit-test point handles first, then bbox areas. Start drag if hit."""
        if self._annotations is None:
            return

        # Keypoints
        for j, item in enumerate(self._kpt_items):
            if not item.isVisible():
                continue
            kpt_idx = item.hit_test_keypoint(scene_pos)
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

        # Segmentation points
        for j, item in enumerate(self._seg_items):
            if not item.isVisible():
                continue
            pt_idx = item.hit_test_point(scene_pos)
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

        # BBox area click → select instance, no drag
        for j, item in enumerate(self._bbox_items):
            if not item.isVisible():
                continue
            if item.boundingRect().contains(scene_pos):
                inst_idx = self._bbox_inst_map[j]
                self._select_instance(inst_idx)
                self._selected_kpt_idx = None
                self._selected_pt_idx = None
                self.instance_selected.emit(self._annotations, inst_idx)
                return

        # Clicked empty space → deselect
        self._deselect()

    def _select_instance(self, inst_idx: int) -> None:
        """Set selected instance and update overlay visibility."""
        # Clear previous point selections
        for item in self._kpt_items:
            item.select_keypoint(None)
        for item in self._seg_items:
            item.select_point(None)
        self._selected_instance = inst_idx
        self._apply_selection_visibility()

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
            inst.keypoints[self._selected_kpt_idx] = Keypoint.from_pixel(
                scene_pos.x(), scene_pos.y(), self._img_w, self._img_h, vis
            )
            # Update only the affected kpt overlay
            for j, item in enumerate(self._kpt_items):
                if self._kpt_inst_map[j] == self._selected_instance:
                    item.update_keypoints(inst.keypoints)
                    item.select_keypoint(self._selected_kpt_idx)
                    break

        elif self._drag_type == "seg" and self._selected_pt_idx is not None:
            if not (inst.mask and self._selected_pt_idx < len(inst.mask.points)):
                return
            if not self._drag_undo_pushed:
                self._undo.push(self._annotations)
                self._drag_undo_pushed = True
            nx = max(0.0, min(1.0, scene_pos.x() / self._img_w))
            ny = max(0.0, min(1.0, scene_pos.y() / self._img_h))
            inst.mask.update_point(self._selected_pt_idx, nx, ny)
            for j, item in enumerate(self._seg_items):
                if self._seg_inst_map[j] == self._selected_instance:
                    item.update_mask(inst.mask)
                    item.select_point(self._selected_pt_idx)
                    break

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
            inst.keypoints[self._selected_kpt_idx] = Keypoint.from_pixel(
                kp.x * self._img_w + dx,
                kp.y * self._img_h + dy,
                self._img_w,
                self._img_h,
                kp.visibility,
            )
            for j, item in enumerate(self._kpt_items):
                if self._kpt_inst_map[j] == self._selected_instance:
                    item.update_keypoints(inst.keypoints)
                    break
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
            inst.mask.update_point(
                self._selected_pt_idx,
                max(0.0, min(1.0, x + dx)),
                max(0.0, min(1.0, y + dy)),
            )
            for j, item in enumerate(self._seg_items):
                if self._seg_inst_map[j] == self._selected_instance:
                    item.update_mask(inst.mask)
                    break
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
                for j, item in enumerate(self._kpt_items):
                    if self._kpt_inst_map[j] == self._selected_instance:
                        item.update_keypoints(inst.keypoints)
                        break
                self._selected_kpt_idx = None
                self._mark_dirty()
                return True

        if self._selected_pt_idx is not None:
            if inst.mask:
                self._undo.push(self._annotations)
                inst.mask.remove_point(self._selected_pt_idx)
                for j, item in enumerate(self._seg_items):
                    if self._seg_inst_map[j] == self._selected_instance:
                        item.update_mask(inst.mask)
                        break
                self._selected_pt_idx = None
                self._mark_dirty()
                return True

        return False
