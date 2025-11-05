from typing import Optional, List

import numpy as np

import napari
from qtpy import QtWidgets
from qtpy.QtCore import QTimer
from magicgui.widgets import Widget, Container, FunctionGui

from . import util as vutil
from . import _widgets as widgets
from ._state import AnnotatorState


class _AnnotatorBase(QtWidgets.QScrollArea):
    """Base class for micro_sam annotation plugins.

    Implements the logic for the 2d, 3d and tracking annotator.
    The annotators differ in their data dimensionality and the widgets.
    """

    def _require_layers(self, layer_choices: Optional[List[str]] = None):

        # Check whether the image is initialized already. And use the image shape and scale for the layers.
        state = AnnotatorState()
        shape = self._shape if state.image_shape is None else state.image_shape

        # Add the label layers for the current object, the automatic segmentation and the committed segmentation.
        # If the annotator is 4D (self._is_4d == True) then create persistent 4D containers
        # and do not create the per-timestep 3D layers. Otherwise fall back to 3D layers.
        image_scale = state.image_scale
        is_4d = getattr(self, "_is_4d", False)

        if is_4d:
            # dummy 4D data: (T, Z, Y, X) with T=1 as placeholder
            dummy_data = np.zeros((1,) + tuple(self._shape), dtype="uint32")

            # current_object_4d
            if "current_object_4d" not in self._viewer.layers:
                if layer_choices and "current_object" in layer_choices:
                    widgets._validation_window_for_missing_layer("current_object")
                self._viewer.add_labels(data=dummy_data, name="current_object_4d")
                if image_scale is not None:
                    try:
                        self._viewer.layers["current_object_4d"].scale = (1.0,) + tuple(image_scale)
                    except Exception:
                        pass

            # auto_segmentation_4d
            if "auto_segmentation_4d" not in self._viewer.layers:
                if layer_choices and "auto_segmentation" in layer_choices:
                    widgets._validation_window_for_missing_layer("auto_segmentation")
                self._viewer.add_labels(data=dummy_data, name="auto_segmentation_4d")
                if image_scale is not None:
                    try:
                        self._viewer.layers["auto_segmentation_4d"].scale = (1.0,) + tuple(image_scale)
                    except Exception:
                        pass

            # committed_objects_4d
            if "committed_objects_4d" not in self._viewer.layers:
                if layer_choices and "committed_objects" in layer_choices:
                    widgets._validation_window_for_missing_layer("committed_objects")
                self._viewer.add_labels(data=dummy_data, name="committed_objects_4d")
                # Randomize colors so it is easy to see when object committed.
                try:
                    self._viewer.layers["committed_objects_4d"].new_colormap()
                except Exception:
                    pass
                if image_scale is not None:
                    try:
                        self._viewer.layers["committed_objects_4d"].scale = (1.0,) + tuple(image_scale)
                    except Exception:
                        pass

        else:
            # fallback: create per-timestep 3D label layers
            dummy_data = np.zeros(shape, dtype="uint32")

            if "current_object" not in self._viewer.layers:
                if layer_choices and "current_object" in layer_choices:  # Check at 'commit' call button.
                    widgets._validation_window_for_missing_layer("current_object")
                self._viewer.add_labels(data=dummy_data, name="current_object")
                if image_scale is not None:
                    try:
                        self._viewer.layers["current_objects"].scale = image_scale
                    except Exception:
                        pass

            if "auto_segmentation" not in self._viewer.layers:
                if layer_choices and "auto_segmentation" in layer_choices:  # Check at 'commit' call button.
                    widgets._validation_window_for_missing_layer("auto_segmentation")
                self._viewer.add_labels(data=dummy_data, name="auto_segmentation")
                if image_scale is not None:
                    try:
                        self._viewer.layers["auto_segmentation"].scale = image_scale
                    except Exception:
                        pass

            if "committed_objects" not in self._viewer.layers:
                if layer_choices and "committed_objects" in layer_choices:  # Check at 'commit' call button.
                    widgets._validation_window_for_missing_layer("committed_objects")
                self._viewer.add_labels(data=dummy_data, name="committed_objects")
                # Randomize colors so it is easy to see when object committed.
                self._viewer.layers["committed_objects"].new_colormap()
                if image_scale is not None:
                    try:
                        self._viewer.layers["committed_objects"].scale = image_scale
                    except Exception:
                        pass

        # Add the point layer for point prompts.
        self._point_labels = ["positive", "negative"]
        if "point_prompts" in self._viewer.layers:
            self._point_prompt_layer = self._viewer.layers["point_prompts"]
        else:
            self._point_prompt_layer = self._viewer.add_points(
                name="point_prompts",
                property_choices={"label": self._point_labels},
                border_color="label",
                border_color_cycle=vutil.LABEL_COLOR_CYCLE,
                symbol="o",
                face_color="transparent",
                border_width=0.5,
                size=12,
                ndim=self._ndim,
            )
            self._point_prompt_layer.border_color_mode = "cycle"

        if "prompts" not in self._viewer.layers:
            # Add the shape layer for box and other shape prompts.
            self._viewer.add_shapes(
                face_color="transparent", edge_color="green", edge_width=4, name="prompts", ndim=self._ndim,
            )

    # --- 4D helper utilities -------------------------------------------------
    def _layer4d_exists(self, base_name: str) -> bool:
        """Return True if a 4D layer named '<base_name>_4d' exists in the viewer."""
        return f"{base_name}_4d" in self._viewer.layers

    def get_layer_data(self, name: str):
        """Return the numpy data for a layer. If a 4D layer exists and the annotator
        exposes `current_timestep`, return the 3D slice for that timestep.
        """
        layer4_name = f"{name}_4d"
        if layer4_name in self._viewer.layers and hasattr(self, "current_timestep"):
            layer4 = self._viewer.layers[layer4_name]
            return layer4.data[self.current_timestep]
        if name in self._viewer.layers:
            return self._viewer.layers[name].data
        raise KeyError(f"Layer {name} not found")

    def set_layer_data(self, name: str, data):
        """Set data for a layer. If a 4D layer exists and `current_timestep` is present,
        write into the correct timestep slice and update any 3D per-timestep layer as well.
        """
        layer4_name = f"{name}_4d"
        if layer4_name in self._viewer.layers and hasattr(self, "current_timestep"):
            layer4 = self._viewer.layers[layer4_name]
            # assign into the slice
            layer4.data[self.current_timestep] = data
            try:
                layer4.refresh()
            except Exception:
                pass
            # keep per-timestep 3D layer in sync if present
            if name in self._viewer.layers:
                try:
                    self._viewer.layers[name].data = data
                    self._viewer.layers[name].refresh()
                except Exception:
                    pass
            return

        # fallback: set the 3D layer directly
        if name in self._viewer.layers:
            self._viewer.layers[name].data = data
            try:
                self._viewer.layers[name].refresh()
            except Exception:
                pass
            return

        raise KeyError(f"Layer {name} not found")

    def set_layer_scale(self, name: str, scale):
        """Set layer scale for both 4D container (if present) and per-timestep 3D layer."""
        layer4_name = f"{name}_4d"
        if layer4_name in self._viewer.layers:
            try:
                self._viewer.layers[layer4_name].scale = scale
            except Exception:
                pass
        if name in self._viewer.layers:
            try:
                self._viewer.layers[name].scale = scale
            except Exception:
                pass

    # Child classes have to implement this function and create a dictionary with the widgets.
    def _get_widgets(self):
        raise NotImplementedError("The child classes of _AnnotatorBase have to implement _get_widgets.")

    def _create_widgets(self):
        # Create the embedding widget and connect all events related to it.
        self._embedding_widget = widgets.EmbeddingWidget()
        # Connect events for the image selection box.
        self._viewer.layers.events.inserted.connect(self._embedding_widget.image_selection.reset_choices)
        self._viewer.layers.events.removed.connect(self._embedding_widget.image_selection.reset_choices)
        # Connect the run button with the function to update the image.
        self._embedding_widget.run_button.clicked.connect(self._update_image)

        # Create the prompt widget. (The same for all plugins.)
        self._prompt_widget = widgets.create_prompt_menu(self._point_prompt_layer, self._point_labels)

        # Create the dictionary for the widgets and get the widgets of the child plugin.
        self._widgets = {"embeddings": self._embedding_widget, "prompts": self._prompt_widget}
        self._widgets.update(self._get_widgets())
        # Create timestep controls (prev/next/goto and edit-mode toggle)
        # placed here so they are initialized when widgets are created
        self._create_timestep_controls()

    def _create_keybindings(self):
        @self._viewer.bind_key("s", overwrite=True)
        def _segment(viewer):
            self._widgets["segment"](viewer)

        # Note: we also need to over-write the keybindings for specific layers.
        # See https://github.com/napari/napari/issues/7302 for details.
        # Here, we need to over-write the 's' keybinding for both of the prompt layers.
        prompt_layer = self._viewer.layers["prompts"]
        point_prompt_layer = self._viewer.layers["point_prompts"]

        @prompt_layer.bind_key("s", overwrite=True)
        def _segment_prompts(event):
            self._widgets["segment"](self._viewer)

        @point_prompt_layer.bind_key("s", overwrite=True)
        def _segment_point_prompts(event):
            self._widgets["segment"](self._viewer)

        @self._viewer.bind_key("c", overwrite=True)
        def _commit(viewer):
            self._widgets["commit"](viewer)

        @self._viewer.bind_key("t", overwrite=True)
        def _toggle_label(event=None):
            vutil.toggle_label(self._point_prompt_layer)

        @self._viewer.bind_key("Shift-C", overwrite=True)
        def _clear_annotations(viewer):
            self._widgets["clear"](viewer)

        if "segment_nd" in self._widgets:
            @self._viewer.bind_key("Shift-S", overwrite=True)
            def _seg_nd(viewer):
                self._widgets["segment_nd"]()

    # --- timestep UI -------------------------------------------------------
    def _create_timestep_controls(self):
        """Add Prev/Next/Goto controls and an edit-mode toggle to avoid accidental timestep changes."""
        # A small horizontal widget at the top of the annotator
        control_widget = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout()
        control_widget.setLayout(hl)

        prev_btn = QtWidgets.QPushButton("◀ Prev")
        next_btn = QtWidgets.QPushButton("Next ▶")
        spin = QtWidgets.QSpinBox()
        spin.setMinimum(0)
        spin.setMaximum(0)
        spin.setPrefix("t=")
        goto_btn = QtWidgets.QPushButton("Go")
        allow_toggle = QtWidgets.QCheckBox("Allow timestep changes")
        allow_toggle.setChecked(True)

        hl.addWidget(prev_btn)
        hl.addWidget(next_btn)
        hl.addWidget(spin)
        hl.addWidget(goto_btn)
        hl.addWidget(allow_toggle)

        # store references
        self._timestep_control_widget = control_widget
        self._timestep_prev_btn = prev_btn
        self._timestep_next_btn = next_btn
        self._timestep_spinbox = spin
        self._timestep_goto_btn = goto_btn
        self._timestep_allow_toggle = allow_toggle

        # initial enabled state
        prev_btn.setEnabled(False)
        next_btn.setEnabled(False)
        goto_btn.setEnabled(False)
        spin.setEnabled(False)

        def _refresh_enabled():
            enabled = allow_toggle.isChecked()
            self._timestep_prev_btn.setEnabled(enabled and getattr(AnnotatorState().annotator, "previous_timestep", None) is not None)
            self._timestep_next_btn.setEnabled(enabled and getattr(AnnotatorState().annotator, "next_timestep", None) is not None)
            self._timestep_goto_btn.setEnabled(enabled and getattr(AnnotatorState().annotator, "_load_timestep", None) is not None)
            self._timestep_spinbox.setEnabled(enabled)

        allow_toggle.stateChanged.connect(lambda _: _refresh_enabled())

        # When the napari timeslider changes (e.g. user drags time), load the corresponding timestep
        # debounce timer to avoid rapid reloads while dragging the timeslider
        self._timestep_change_timer = QTimer()
        self._timestep_change_timer.setSingleShot(True)
        self._pending_timestep = None

        def _apply_pending_timestep():
            t = self._pending_timestep
            self._pending_timestep = None
            if t is None:
                return
            try:
                annotator = AnnotatorState().annotator
                if annotator is None:
                    return
                loader = getattr(annotator, "_load_timestep", None)
                if loader is not None:
                    loader(int(t))
            except Exception:
                pass

        self._timestep_change_timer.timeout.connect(_apply_pending_timestep)

        def _on_dims_change(event=None):
            # Only respond when allowed
            try:
                if not self._timestep_allow_toggle.isChecked():
                    return
            except Exception:
                pass

            try:
                step = tuple(self._viewer.dims.current_step)
                if len(step) > 0:
                    t = int(step[0])
                else:
                    return
            except Exception:
                return

            # store pending timestep and restart debounce timer
            self._pending_timestep = t
            # if annotator supports a fast preview hook, call it immediately for snappy feedback
            try:
                annotator = AnnotatorState().annotator
                if annotator is not None and hasattr(annotator, "_preview_timestep"):
                    try:
                        annotator._preview_timestep(t)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                self._timestep_change_timer.start(250)
            except Exception:
                # fallback: directly apply if timer fails
                _apply_pending_timestep()

        

        try:
            # connect to dims current_step changes
            self._viewer.dims.events.current_step.connect(_on_dims_change)
        except Exception:
            # older napari versions or missing event API - ignore silently
            pass

        def _on_prev():
            annotator = AnnotatorState().annotator
            if annotator is None:
                return
            fn = getattr(annotator, "previous_timestep", None)
            if fn is not None:
                fn()
                # update spinbox value if available
                if hasattr(annotator, "current_timestep"):
                    try:
                        self._timestep_spinbox.setValue(int(annotator.current_timestep))
                    except Exception:
                        pass

        def _on_next():
            annotator = AnnotatorState().annotator
            if annotator is None:
                return
            fn = getattr(annotator, "next_timestep", None)
            if fn is not None:
                fn()
                if hasattr(annotator, "current_timestep"):
                    try:
                        self._timestep_spinbox.setValue(int(annotator.current_timestep))
                    except Exception:
                        pass

        def _on_goto():
            annotator = AnnotatorState().annotator
            if annotator is None:
                return
            go_fn = getattr(annotator, "_load_timestep", None)
            idx = int(self._timestep_spinbox.value())
            if go_fn is not None:
                try:
                    go_fn(idx)
                except Exception:
                    # fallback: try to set current_timestep then call _load_timestep
                    try:
                        setattr(annotator, "current_timestep", idx)
                        go_fn(idx)
                    except Exception:
                        pass
            else:
                # if no explicit loader, try to set attribute and hope viewer updates
                try:
                    setattr(annotator, "current_timestep", idx)
                except Exception:
                    pass

        prev_btn.clicked.connect(lambda _: _on_prev())
        next_btn.clicked.connect(lambda _: _on_next())
        goto_btn.clicked.connect(lambda _: _on_goto())

        # Insert at the top of the annotator layout so controls are visible first
        self._annotator_widget.layout().insertWidget(0, control_widget)

        # refresh enabled once (in case an annotator already exists)
        _refresh_enabled()

    # We could implement a better way of initializing the segmentation result,
    # so that instead of just passing a numpy array an existing layer from the napari
    # viewer can be chosen.
    # See https://github.com/computational-cell-analytics/micro-sam/issues/335
    def __init__(self, viewer: "napari.viewer.Viewer", ndim: int) -> None:
        """Create the annotator GUI.

        Args:
            viewer: The napari viewer.
            ndim: The number of spatial dimension of the image data (2 or 3).
        """
        super().__init__()
        self._viewer = viewer
        self._annotator_widget = QtWidgets.QWidget()
        self._annotator_widget.setLayout(QtWidgets.QVBoxLayout())

        # Add the layers for prompts and segmented obejcts.
        # Initialize with a dummy shape, which is reset to the correct shape once an image is set.
        self._ndim = ndim
        self._shape = (256, 256) if ndim == 2 else (16, 256, 256)
        self._require_layers()

        # Create all the widgets and add them to the layout.
        self._create_widgets()
        for widget in self._widgets.values():
            widget_frame = QtWidgets.QGroupBox()
            widget_layout = QtWidgets.QVBoxLayout()
            if isinstance(widget, (Container, FunctionGui, Widget)):
                # This is a magicgui type and we need to get the native qt widget.
                widget_layout.addWidget(widget.native)
            else:
                # This is a qt type and we add the widget directly.
                widget_layout.addWidget(widget)
            widget_frame.setLayout(widget_layout)
            self._annotator_widget.layout().addWidget(widget_frame)

        # Add the widgets to the state.
        AnnotatorState().widgets = self._widgets

        # Add the key bindings in common between all annotators.
        self._create_keybindings()

        # Add the widget to the scroll area.
        self.setWidgetResizable(True)  # Allow widget to resize within scroll area.
        self.setWidget(self._annotator_widget)

    def _update_image(self, segmentation_result=None):
        state = AnnotatorState()

        # Whether embeddings already exist and avoid clearing objects in layers.
        if state.skip_recomputing_embeddings:
            return

        # This is encountered when there is no image layer available / selected.
        # In this case, we need not update the image shape or check for changes.
        # NOTE: On code-level, this happens when '__init__' method is called by '_AnnotatorBase',
        #       where one of the first steps is to '_create_widgets', which reaches here.
        if state.image_shape is None:
            return

        # Update the image shape if it has changed.
        if state.image_shape != self._shape:
            if len(state.image_shape) != self._ndim:
                raise RuntimeError(
                    f"The dim of the annotator {self._ndim} does not match the image data of shape {state.image_shape}."
                )
            self._shape = state.image_shape

        # Before we reset the layers, we ensure all expected layers exist.
        self._require_layers()

        # Update the image scale.
        scale = state.image_scale

        # Reset all layers.
        self.set_layer_data("current_object", np.zeros(self._shape, dtype="uint32"))
        self.set_layer_scale("current_object", scale)
        self.set_layer_data("auto_segmentation", np.zeros(self._shape, dtype="uint32"))
        self.set_layer_scale("auto_segmentation", scale)

        if segmentation_result is None or segmentation_result is False:
            self.set_layer_data("committed_objects", np.zeros(self._shape, dtype="uint32"))
        else:
            assert segmentation_result.shape == self._shape
            self.set_layer_data("committed_objects", segmentation_result)
        self.set_layer_scale("committed_objects", scale)

        self.set_layer_scale("point_prompts", scale)
        self.set_layer_scale("prompts", scale)

        vutil.clear_annotations(self._viewer, clear_segmentations=False)

        # Update timestep controls if a 4D annotator is present
        try:
            self._update_timestep_controls()
        except Exception:
            pass

    def _update_timestep_controls(self):
        """Refresh the timestep spinbox range and current value based on the annotator."""
        try:
            annotator = AnnotatorState().annotator
        except Exception:
            annotator = None

        if annotator is None:
            # disable controls
            try:
                self._timestep_spinbox.setMaximum(0)
                self._timestep_spinbox.setValue(0)
                self._timestep_prev_btn.setEnabled(False)
                self._timestep_next_btn.setEnabled(False)
                self._timestep_goto_btn.setEnabled(False)
            except Exception:
                pass
            return

        # if annotator exposes n_timesteps use it
        n = getattr(annotator, "n_timesteps", None)
        ct = getattr(annotator, "current_timestep", None)
        try:
            if n is None:
                self._timestep_spinbox.setMaximum(0)
            else:
                self._timestep_spinbox.setMaximum(max(0, int(n) - 1))
            if ct is not None:
                self._timestep_spinbox.setValue(int(ct))
            # enable controls only if toggle allows it
            allow = getattr(self, "_timestep_allow_toggle", None)
            allow_ok = True if allow is None else allow.isChecked()
            self._timestep_prev_btn.setEnabled(allow_ok and ct is not None and int(ct) > 0)
            self._timestep_next_btn.setEnabled(allow_ok and n is not None and int(ct) < int(n) - 1)
            self._timestep_goto_btn.setEnabled(allow_ok)
        except Exception:
            pass
