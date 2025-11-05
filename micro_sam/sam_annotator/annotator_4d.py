import numpy as np
import threading
from qtpy import QtWidgets
from napari.utils.notifications import show_info
from micro_sam.sam_annotator.annotator_3d import Annotator3d
from micro_sam.sam_annotator._state import AnnotatorState
from pathlib import Path
import json
from micro_sam import instance_segmentation
from .util import _load_amg_state, _load_is_state
from . import util as _vutil
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation
from skimage.transform import resize as _sk_resize


def _select_array_from_zarr_group(f):
    """Pick a zarr.Array-like child from a zarr.Group.

    Preference order:
      - dataset named 'features'
      - common alternate names
      - first array-like child at depth 1
      - first array-like child inside a top-level group at depth 2

    Returns the array-like object or None if none found.
    """
    try:
        # Prefer explicit 'features'
        if "features" in f:
            return f["features"]
    except Exception:
        pass

    # common alternative names
    for alt in ("feats", "features_0", "features0", "arr", "data"):
        try:
            if alt in f:
                return f[alt]
        except Exception:
            pass

    # first pass: find first array-like child at depth 1
    try:
        for name, obj in f.items():
            try:
                if hasattr(obj, "ndim") or hasattr(obj, "shape"):
                    return obj
            except Exception:
                # if obj is a Group, try depth-2
                try:
                    for cname, cobj in obj.items():
                        if hasattr(cobj, "ndim") or hasattr(cobj, "shape"):
                            return cobj
                except Exception:
                    continue
    except Exception:
        pass

    return None



class TimestepEmbeddingManager:
    """Manage lazy loading of per-timestep embeddings stored as zarr files.

    This keeps only one materialized embedding in memory (the most recently
    used timestep). When the timestep changes we load the matching
    `embeddings_t{t}.zarr` lazily (in a background thread) and activate it on
    the parent annotator by calling `_ensure_embeddings_active_for_t(t)`.
    """

    def __init__(self, annotator, embeddings_dir: str | Path | None = None, lazy: bool = True):
        self.annotator = annotator
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir is not None else None
        self.lazy = bool(lazy)
        self._lock = threading.Lock()
        # cache for the currently materialized timestep
        self.cached_t = None
        self.cached_entry = None
        self.cached_path = None

    def set_embeddings_dir(self, path: str | Path):
        self.embeddings_dir = Path(path)

    def get_current_embedding(self):
        return self.cached_entry

    def on_timestep_changed(self, t: int):
        """Callback for timestep changes; triggers loading for timestep `t`."""
        try:
            # schedule load in background to avoid blocking the UI
            thread = threading.Thread(target=self.load_embedding_for_timestep, args=(int(t),), daemon=True)
            thread.start()
        except Exception:
            # best-effort synchronous fallback
            try:
                self.load_embedding_for_timestep(int(t))
            except Exception:
                pass

    def _make_zarr_path_for_t(self, t: int):
        # prefer explicit last dir from annotator
        p = None
        if getattr(self.annotator, "_last_embeddings_dir", None):
            p = Path(self.annotator._last_embeddings_dir)
        if p is None and self.embeddings_dir is not None:
            p = self.embeddings_dir
        if p is None:
            return None
        cand = p / f"embeddings_t{t}.zarr"
        if cand.exists():
            return cand
        # also accept alternative naming
        cand2 = p / f"t{t}.zarr"
        if cand2.exists():
            return cand2
        return None

    def load_embedding_for_timestep(self, t: int):
        """Load (lazily) the embedding for timestep `t` and activate it.

        This will materialize the zarr store (open it) and call
        `annotator._ensure_embeddings_active_for_t(t)` so the global state uses
        the newly-loaded embeddings. Only one embedding is kept materialized;
        the previous one is released (replaced with a path placeholder) if it
        was loaded from disk by this manager.
        """
        with self._lock:
            # If we already have the requested timestep cached, nothing to do
            if self.cached_t == int(t) and self.cached_entry is not None:
                return self.cached_entry

            # If annotator already knows about an embedding entry for t, prefer that
            try:
                existing = self.annotator.embeddings_4d.get(int(t))
                if existing is not None and isinstance(existing, dict) and "features" in existing:
                    # materialized already by other code; adopt as cache
                    self._release_cached_if_needed(exclude_t=int(t))
                    self.cached_t = int(t)
                    self.cached_entry = existing
                    self.cached_path = existing.get("path") if isinstance(existing, dict) else None
                    # ensure annotator binds it
                    try:
                        self.annotator._ensure_embeddings_active_for_t(int(t))
                    except Exception:
                        pass
                    return self.cached_entry
            except Exception:
                pass

            # Try to resolve zarr path
            zpath = self._make_zarr_path_for_t(int(t))
            if zpath is None:
                # Fallback: if annotator.embeddings_4d contains an entry (e.g., mapping produced by set_embeddings_folder), use it
                entry = self.annotator.embeddings_4d.get(int(t))
                if entry is not None:
                    # trigger annotator's activation path (it will materialize if needed)
                    try:
                        self.annotator._ensure_embeddings_active_for_t(int(t))
                    except Exception:
                        pass
                    return entry
                return None

            # Mark annotator mapping with a lazy path so other parts can see it
            try:
                self.annotator.embeddings_4d[int(t)] = {"path": str(zpath)}
            except Exception:
                pass

            # Materialize (open) the zarr in background thread context (we're already in a thread)
            try:
                import zarr as _zarr
                try:
                    show_info(f"Loading embeddings for timestep {t} from {zpath}...")
                except Exception:
                    pass
                f = _zarr.open(str(zpath), mode="r")
                # Pick a suitable array-like dataset from the zarr group.
                feats = _select_array_from_zarr_group(f)
                if feats is None:
                    try:
                        show_info(f"No suitable array dataset found inside {zpath}; cannot load embeddings for timestep {t}.")
                    except Exception:
                        pass
                    return None
                attrs = getattr(feats, "attrs", {}) or {}
                input_size = attrs.get("input_size")
                original_size = attrs.get("original_size")
                if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                    try:
                        inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                        input_size = input_size or inferred
                        original_size = original_size or inferred
                    except Exception:
                        input_size = input_size or None
                        original_size = original_size or None

                entry = {"features": feats, "input_size": input_size, "original_size": original_size, "path": str(zpath)}

                # release old cached if it was loaded by manager
                self._release_cached_if_needed()

                # cache this one
                self.cached_t = int(t)
                self.cached_entry = entry
                self.cached_path = str(zpath)

                # store in annotator mapping and activate
                try:
                    self.annotator.embeddings_4d[int(t)] = entry
                except Exception:
                    pass
                try:
                    # call annotator activation so AnnotatorState binds this embedding
                    self.annotator._ensure_embeddings_active_for_t(int(t))
                except Exception:
                    pass
                try:
                    show_info(f"Embeddings for timestep {t} loaded.")
                except Exception:
                    pass
                return entry
            except Exception:
                try:
                    show_info(f"Failed to load embeddings for timestep {t} from {zpath}")
                except Exception:
                    pass
                return None

    def _release_cached_if_needed(self, exclude_t: int | None = None):
        """Release the currently cached embedding if it was loaded from disk by this manager.

        If the previous entry was materialized from a zarr on-disk store, replace
        it in `annotator.embeddings_4d` with a lightweight {'path': ...} mapping
        so memory can be reclaimed.
        """
        try:
            if self.cached_t is None:
                return
            if exclude_t is not None and int(self.cached_t) == int(exclude_t):
                return
            # Only replace if we have a cached_path (i.e., loaded from disk)
            if self.cached_path and self.cached_t is not None:
                try:
                    # replace materialized entry with a path-only placeholder
                    self.annotator.embeddings_4d[int(self.cached_t)] = {"path": str(self.cached_path)}
                except Exception:
                    try:
                        del self.annotator.embeddings_4d[int(self.cached_t)]
                    except Exception:
                        pass
            # clear local references so zarr can be freed by GC
            self.cached_t = None
            self.cached_entry = None
            self.cached_path = None
        except Exception:
            pass



class MicroSAM4DAnnotator(Annotator3d):
    """
    4D annotator for (T, Z, Y, X) time-series data.

    This class keeps a persistent 4D image layer (`raw_4d`) and a persistent
    4D labels layer (`committed_objects_4d`). Per-timestep interactive 3D
    layers (editable views) are created once and updated in-place when the
    Napari time slider (dims.current_step[0]) changes. Committing a
    segmentation writes the result back into the 4D label array and refreshes
    the visible 3D view.
    """

    def __init__(self, viewer):
        # mark this instance as a 4D annotator so base class will create _4d
        # container layers instead of per-timestep 3D layers during init
        try:
            # set flag before calling base init so _AnnotatorBase can detect 4D
            self._is_4d = True
        except Exception:
            pass
        super().__init__(viewer)
        self.current_timestep = 0
        self.image_4d = None
        self.use_preview = True
        # 4D arrays (T, Z, Y, X)
        self.segmentation_4d = None
        self.auto_segmentation_4d = None
        self.current_object_4d = None
        self.point_prompts_4d = None
        self.n_timesteps = 0
        # small per-timestep cache (optional)
        self._segmentation_cache = None
        # per-timestep embeddings cache: mapping t -> embedding dict or lazy entry
        self.embeddings_4d = {}
        # flags for background materialization (t -> bool)
        self._embedding_loading = {}
        # currently-active timestep whose embeddings are bound to AnnotatorState
        self._active_embedding_t = None

        # remember last directory where embeddings were saved/loaded
        self._last_embeddings_dir = None

        # Timestep embedding manager for lazy per-timestep zarr loading
        try:
            self.timestep_embedding_manager = TimestepEmbeddingManager(self)
        except Exception:
            self.timestep_embedding_manager = None

                # Add small embedding controls to the annotator dock (Compute embeddings current/all T)
        try:
            emb_widget = QtWidgets.QWidget()
            emb_layout = QtWidgets.QVBoxLayout()
            emb_widget.setLayout(emb_layout)

            # Row with two compute buttons
            row = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout()
            row.setLayout(row_layout)
            btn_current = QtWidgets.QPushButton("Compute embeddings (current T)")
            btn_all = QtWidgets.QPushButton("Compute embeddings (all T)")
            row_layout.addWidget(btn_current)
            row_layout.addWidget(btn_all)

            # Only add the compute buttons row
            emb_layout.addWidget(row)

            # Add ID remapper widget
            try:
                from ._widgets import IdRemapperWidget
                remapper = IdRemapperWidget(self)
                emb_layout.addWidget(remapper)
            except Exception:
                pass
            # Removed timestep list widget and activation controls

            def _compute_current():
                try:
                    t = int(getattr(self, "current_timestep", 0) or 0)
                    show_info(f"Computing embeddings for timestep {t} — this may take a while.")
                    self.compute_embeddings_for_timestep(t)
                    show_info("Embeddings computed for current timestep.")
                except Exception as e:
                    print(f"Failed to compute embeddings for timestep {t}: {e}")

            def _compute_all():
                try:
                    show_info("Computing embeddings for all timesteps — this may take a long time.")
                    self.compute_embeddings_for_all_timesteps()
                    show_info("Embeddings computed for all timesteps.")
                except Exception as e:
                    print(f"Failed to compute embeddings for all timesteps: {e}")

            btn_current.clicked.connect(lambda _: _compute_current())
            btn_all.clicked.connect(lambda _: _compute_all())

            # Removed save/load embeddings folder functions and handlers

            # Insert the embedding widget at the top of the annotator layout
            try:
                self._annotator_widget.layout().insertWidget(0, emb_widget)
            except Exception:
                # fallback: add at end
                try:
                    self._annotator_widget.layout().addWidget(emb_widget)
                except Exception:
                    pass
        except Exception:
            # don't fail initialization if Qt isn't available
            pass


    def _reorder_layers(self):
        """Ensure 'raw_4d' and optional 'raw' are the bottom-most layers."""
        try:
            if "raw_4d" in self._viewer.layers:
                # move raw_4d to bottom
                self._viewer.layers.move("raw_4d", 0)
            if "raw" in self._viewer.layers:
                idx = 1 if "raw_4d" in self._viewer.layers else 0
                self._viewer.layers.move("raw", idx)
        except Exception:
            # best effort; do not fail
            pass

            # DROPDOWN REMAPPER WIDGET IF YOU WANT IT BACK
            # # Create and add the ID remapper widget if not already added
            # if not hasattr(self, '_remapper_widget'):
            #     try:
            #         from ._widgets import IdRemapperWidget
            #         from superqt import QCollapsible
            #         remapper = IdRemapperWidget(self)
            #         remapper_widget = QtWidgets.QWidget()
            #         remapper_widget.setLayout(QtWidgets.QVBoxLayout())
            #         remapper_collapsible = QCollapsible("ID Remapper", remapper_widget)
            #         remapper_collapsible.addWidget(remapper)
            #         remapper_widget.layout().addWidget(remapper_collapsible)
            #         if hasattr(self, '_annotator_widget') and hasattr(self._annotator_widget, 'layout'):
            #             self._annotator_widget.layout().insertWidget(1, remapper_widget)
            #         self._remapper_widget = remapper_widget
            #     except Exception as e:
            #         print(f"Failed to create ID remapper widget: {str(e)}")

    def update_image(self, image_4d):
        """Initialize annotator state with a 4D image.

        Adds/updates `raw_4d` and `committed_objects_4d` (4D labels). Also
        creates persistent 3D interactive layers for the current timestep
        that will be updated in-place when switching timesteps.
        """
        if image_4d.ndim != 4:
            raise ValueError(f"Expected 4D data (T,Z,Y,X), got {image_4d.shape}")

        self.image_4d = image_4d
        self.n_timesteps = image_4d.shape[0]
        self.current_timestep = 0

        # configure napari dims for time-series
        try:
            self._viewer.dims.ndim = 4
            self._viewer.dims.axis_labels = ["T", "Z", "Y", "X"]
        except Exception:
            pass
        # add or update persistent 4D raw image layer
        if "raw_4d" in self._viewer.layers:
            try:
                self._viewer.layers["raw_4d"].data = image_4d
                self._viewer.layers["raw_4d"].visible = True
            except Exception:
                pass
        else:
            try:
                self._viewer.add_image(image_4d, name="raw_4d")
            except Exception:
                pass

        # initialize persistent 4D containers and ensure Napari layers reference
        # the same underlying arrays (so edits in Napari mutate our arrays)
        self.segmentation_4d = np.zeros_like(image_4d, dtype=np.uint32)
        self.auto_segmentation_4d = np.zeros_like(image_4d, dtype=np.uint32)
        self.current_object_4d = np.zeros_like(image_4d, dtype=np.uint32)
        self.point_prompts_4d = [None] * self.n_timesteps
        self._segmentation_cache = [self.segmentation_4d[t].copy() for t in range(self.n_timesteps)]

        # add or update persistent 4D labels layers so Napari edits directly
        # mutate the underlying 4D arrays (no per-timestep recreation)
        try:
            if "committed_objects_4d" in self._viewer.layers:
                self._viewer.layers["committed_objects_4d"].data = self.segmentation_4d
            else:
                self._viewer.add_labels(data=self.segmentation_4d, name="committed_objects_4d")
        except Exception:
            pass

        try:
            if "current_object_4d" in self._viewer.layers:
                self._viewer.layers["current_object_4d"].data = self.current_object_4d
            else:
                self._viewer.add_labels(data=self.current_object_4d, name="current_object_4d")
        except Exception:
            pass

        try:
            if "auto_segmentation_4d" in self._viewer.layers:
                self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
            else:
                self._viewer.add_labels(data=self.auto_segmentation_4d, name="auto_segmentation_4d")
        except Exception:
            pass

        # single persistent points layer for prompts (we update .data in-place)
        try:
            pts0 = np.array(self.point_prompts_4d[0]) if self.point_prompts_4d[0] is not None else np.empty((0, 3))
            if "point_prompts" in self._viewer.layers:
                self._viewer.layers["point_prompts"].data = pts0
            else:
                self._viewer.add_points(pts0, name="point_prompts")
        except Exception:
            pass

        # ensure raw_4d is bottom-most layer
        self._reorder_layers()

        # ensure our local arrays are the same object as Napari layer data
        try:
            if "committed_objects_4d" in self._viewer.layers:
                self.segmentation_4d = self._viewer.layers["committed_objects_4d"].data
            if "current_object_4d" in self._viewer.layers:
                self.current_object_4d = self._viewer.layers["current_object_4d"].data
            if "auto_segmentation_4d" in self._viewer.layers:
                self.auto_segmentation_4d = self._viewer.layers["auto_segmentation_4d"].data
        except Exception:
            pass

        # set initial visible timestep via Napari dims (no layer recreation)
        try:
            # set dims current step to (t, z, y, x) = (0, 0, 0, 0)
            self._viewer.dims.current_step = (0,) + (0,) * (self._viewer.dims.ndim - 1)
            # connect dims handler once
            if not getattr(self, "_dims_handler_connected", False):
                self._viewer.dims.events.current_step.connect(self._on_dims_current_step)
                self._dims_handler_connected = True
        except Exception:
            pass

        # final UI hook (no-op by default)
        self._update_timestep_controls()
    

    def _load_timestep(self, t: int):
        """Switch visible timestep by updating Napari dims without recreating layers.

        Persist any UI-only state (points) before switching, update Napari's
        dims.current_step to show the requested T slice, and refresh the
        points layer to show prompts for the new timestep.
        """
        new_t = int(t)
        if self.image_4d is None:
            return
        if not (0 <= new_t < self.n_timesteps):
            return

        prev_t = getattr(self, "current_timestep", None)
        # persist current points
        try:
            if prev_t is not None and "point_prompts" in self._viewer.layers:
                try:
                    pts = np.array(self._viewer.layers["point_prompts"].data)
                    self.point_prompts_4d[prev_t] = pts if pts.size else None
                except Exception:
                    pass
        except Exception:
            pass

        # set Napari's current_step to the new timestep while preserving
        # the non-time axes (Z/Y/X) if possible. This prevents resetting
        # the Z slider back to 0 when switching timesteps.
        try:
            # get existing step tuple and preserve its non-time entries
            current = list(self._viewer.dims.current_step)
            ndim = max(4, getattr(self._viewer.dims, "ndim", 4))
            if len(current) < ndim:
                current = current + [0] * (ndim - len(current))
            current[0] = new_t
            self._viewer.dims.current_step = tuple(current)
        except Exception:
            try:
                self._viewer.dims.current_step = (new_t,) + (0,) * (max(4, getattr(self._viewer.dims, "ndim", 4)) - 1)
            except Exception:
                pass

        # refresh points layer to the new timestep (in-place)
        try:
            pts_new = self.point_prompts_4d[new_t]
            lay = self._viewer.layers.get("point_prompts", None)
            if lay is not None:
                lay.data = np.array(pts_new) if pts_new is not None else np.empty((0, 3))
        except Exception:
            pass

        self.current_timestep = new_t

        if hasattr(self, "_ensure_embeddings_active_for_t"):
            try:
                self._ensure_embeddings_active_for_t(t)
                # activation is silent to avoid spamming the console
                # (previously printed activation messages here)
            except Exception:
                # silently ignore activation errors; callers may inspect state
                pass

    def _preview_timestep(self, t: int, downscale=(4, 4, 4)):
        """Optional preview while scrubbing — currently a no-op to avoid
        replacing the persistent raw layers which would re-order layers.
        """
        return

    def commit_segmentation(self, seg_volume: np.ndarray):
        """Save a 3D segmentation slice into the 4D labels layer in-place.

        Write directly into the `committed_objects_4d` Napari labels layer so
        that no layer recreation is necessary. Refresh the layer after write.
        """
        t = self.current_timestep
        if t is None:
            return

        try:
            layer = self._viewer.layers.get("committed_objects_4d", None)
            if layer is None:
                # ensure our 4D container exists
                if self.segmentation_4d is None and self.image_4d is not None:
                    self.segmentation_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
                    try:
                        self._viewer.add_labels(data=self.segmentation_4d, name="committed_objects_4d")
                        layer = self._viewer.layers.get("committed_objects_4d")
                    except Exception:
                        layer = None

            if layer is not None and seg_volume is not None:
                # Ensure the segmentation slice matches the target shape. If not,
                # attempt a nearest-neighbour resize to avoid broadcasting/aliasing.
                try:
                    target_shape = layer.data.shape[1:]
                    if getattr(seg_volume, "shape", None) != target_shape:
                        seg_volume = _sk_resize(
                            seg_volume.astype("float32"), target_shape, order=0, preserve_range=True, anti_aliasing=False
                        ).astype(seg_volume.dtype)
                except Exception:
                    # If resizing fails, continue and let assignment raise if incompatible.
                    pass

                # Prevent event-driven handlers from seeing an intermediate state
                # and avoid possible aliasing by assigning a copy under the event blocker.
                try:
                    ev = getattr(layer, "events", None)
                    if ev is not None and hasattr(ev, "data"):
                        with layer.events.data.blocker():
                            layer.data[t] = seg_volume.copy()
                    else:
                        layer.data[t] = seg_volume.copy()
                except Exception:
                    # Fallback to direct assignment without blocker.
                    layer.data[t] = seg_volume.copy()

                # keep local ref in sync
                self.segmentation_4d = layer.data
                try:
                    layer.refresh()
                except Exception:
                    pass

                # update cache
                try:
                    if self._segmentation_cache is None:
                        self._segmentation_cache = [None] * self.n_timesteps
                    self._segmentation_cache[t] = self.segmentation_4d[t].copy()
                except Exception:
                    pass
        except Exception:
            pass

        print(f"✅ Committed segmentation for timestep {t}")

    def save_current_object_to_4d(self):
        """Ensure `current_object_4d` Napari layer and local array are in sync.

        Editing is expected to happen directly on the 4D `current_object_4d`
        labels layer. This method refreshes the local reference so the
        Python-side array remains the same object as Napari layer.data.
        """
        try:
            if "current_object_4d" in self._viewer.layers:
                self.current_object_4d = self._viewer.layers["current_object_4d"].data
        except Exception:
            pass

    def save_point_prompts(self):
        """Save the point_prompts 3D layer into per-timestep storage."""
        t = self.current_timestep
        if "point_prompts" in self._viewer.layers:
            try:
                pts = np.array(self._viewer.layers["point_prompts"].data)
                self.point_prompts_4d[t] = pts
            except Exception:
                pass

    # ----------------- Embedding helpers for 4D -----------------
    def compute_embeddings_for_timestep(self, t: int, model_type: str = None, device: str | None = None, save_path: str | None = None, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute image embeddings for a single timestep and store them in AnnotatorState.

        This wraps AnnotatorState.initialize_predictor for convenience when working with 4D (T,Z,Y,X)
        data. It extracts the 3D volume at timestep `t` and computes embeddings with ndim=3.
        """
        from ._state import AnnotatorState

        if self.image_4d is None:
            raise RuntimeError("No 4D image loaded")
        if not (0 <= t < self.n_timesteps):
            raise IndexError("t out of range")

        image3d = self.image_4d[int(t)]
        state = AnnotatorState()
        # default model_type if not provided
        model_type = model_type or getattr(state, "predictor", None) and getattr(state.predictor, "model_name", None) or None
        # initialize predictor and compute embeddings for this 3D volume
        state.initialize_predictor(
            image3d,
            model_type=model_type or "vit_b_lm",
            ndim=3,
            save_path=save_path,
            device=device,
            tile_shape=tile_shape,
            halo=halo,
            prefer_decoder=prefer_decoder,
        )
        # Capture the computed embeddings and store them per-timestep only
        try:
            embeds = state.image_embeddings
        except Exception:
            embeds = None
        try:
            if embeds is not None:
                # store only in per-timestep cache
                self.embeddings_4d[int(t)] = embeds
        except Exception:
            pass

        # Only bind the embeddings into the global AnnotatorState if this
        # timestep is currently active; otherwise detach to avoid leaking the
        # embedding globally across timesteps.
        try:
            if int(getattr(self, "current_timestep", 0) or 0) == int(t):
                try:
                    state.image_embeddings = embeds
                    # update active marker
                    self._active_embedding_t = int(t)
                except Exception:
                    pass
            else:
                try:
                    # detach any embeddings from the global state
                    state.image_embeddings = None
                    # do not clear predictor, only the embedding handle
                    if getattr(state, "embedding_path", None) is not None and state.embedding_path == save_path:
                        state.embedding_path = None
                except Exception:
                    pass
        except Exception:
            pass
        # Update the state's image_name so widgets reflect the selection
        try:
            state.image_name = getattr(self._viewer.layers.get("raw_4d"), "name", state.image_name)
        except Exception:
            pass
        # Ensure AnnotatorState has image_shape / image_scale set so downstream
        # segmentation widgets and helpers (which read state.image_shape) work
        # when embeddings are computed via this helper (instead of the embedding widget).
        try:
            # image3d has shape (Z, Y, X)
            state.image_shape = tuple(image3d.shape)
        except Exception:
            pass
        try:
            layer = self._viewer.layers.get("raw_4d", None)
            if layer is not None:
                # Napari image layer scale for raw_4d is (T, Z, Y, X). Use the spatial part.
                scale = getattr(layer, "scale", None)
                if scale is not None and len(scale) >= 4:
                    state.image_scale = tuple(scale[1:])
                elif scale is not None and len(scale) == 3:
                    state.image_scale = tuple(scale)
        except Exception:
            pass

        # return the per-timestep stored embedding (may be None)
        return self.embeddings_4d.get(int(t))

    def compute_embeddings_for_all_timesteps(self, model_type: str = None, device: str | None = None, base_save_path: str | None = None, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute embeddings for every timestep. Saves to separate files if base_save_path is provided.

        WARNING: this can be slow and memory/disk intensive. Use with care.
        Returns a list of image_embeddings objects (one per timestep).
        """
        results = []
        # ensure we have a per-timestep embeddings dict
        if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
            self.embeddings_4d = {}

        for t in range(self.n_timesteps):
            sp = None if base_save_path is None else f"{base_save_path}_t{t}.zarr"
            embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=sp, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
            results.append(embeds)
            try:
                # store embedding for this timestep; keep whatever structure compute returned
                self.embeddings_4d[int(t)] = embeds
            except Exception:
                # best-effort: ignore failures to cache
                pass
        return results

    def compute_and_save_embeddings(self, output_dir: str | Path, model_type: str = None, device: str | None = None, per_timestep: bool = True, overwrite: bool = False, tile_shape=None, halo=None, prefer_decoder: bool = True):
        """Compute embeddings for all timesteps and save into `output_dir`.

        By default this creates one zarr store per timestep named
        `embeddings_t{t}.zarr` inside `output_dir`. If `per_timestep` is
        False the method will attempt to compute embeddings in-memory and
        store a single `embeddings.npz` file (may be large).

        Returns a list of embedding objects (as returned by precompute), and
        writes a `manifest.json` into `output_dir` describing the files.
        """
        outp = Path(output_dir)
        outp.mkdir(parents=True, exist_ok=True)

        manifest = {
            "n_timesteps": int(self.n_timesteps),
            "per_timestep": bool(per_timestep),
            "files": {},
        }

        results = []
        if per_timestep:
            for t in range(self.n_timesteps):
                fname = outp / f"embeddings_t{t}.zarr"
                if fname.exists() and not overwrite:
                    # load existing (skip compute)
                    results.append({"path": str(fname)})
                    manifest["files"][str(t)] = str(fname.name)
                    continue

                # compute and save to zarr
                save_path = str(fname)
                embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=save_path, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
                results.append(embeds)
                manifest["files"][str(t)] = str(fname.name)

        else:
            # compute all embeddings in-memory and write a single npz
            arrs = {}
            for t in range(self.n_timesteps):
                embeds = self.compute_embeddings_for_timestep(t=t, model_type=model_type, device=device, save_path=None, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
                # `embeds` is a dict containing 'features' (numpy or zarr)
                feats = embeds.get("features")
                # If features are zarr-like, read into memory (could be large)
                try:
                    if hasattr(feats, "[:]"):
                        feats_np = feats[:]  # zarr or numpy-like
                    else:
                        feats_np = np.asarray(feats)
                except Exception:
                    feats_np = np.asarray(feats)
                arrs[f"t{t}"] = feats_np
                results.append({"features": feats_np})

            npz_path = outp / "embeddings.npz"
            if npz_path.exists() and not overwrite:
                raise FileExistsError(f"{npz_path} already exists. Use overwrite=True to replace.")
            np.savez_compressed(str(npz_path), **arrs)
            manifest["files"] = {str(t): str(npz_path.name) for t in range(self.n_timesteps)}

        # write manifest
        manifest_path = outp / "manifest.json"
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh, indent=2)

        return results

    def load_saved_embeddings(self, path: str | Path, lazy: bool = True):
        """Load embeddings previously saved with `compute_and_save_embeddings`.

        Args:
            path: Directory containing embeddings (manifest.json) or a single .npz file.
            lazy: If True and the embeddings are stored as zarr stores, return zarr objects
                  without loading full arrays into memory. If False, load numpy arrays.

        Returns:
            A dict mapping timestep (int) -> embedding dict (with at least key 'features').
        """
        p = Path(path)
        results = {}
        if p.is_dir():
            # Detect a parent directory containing per-timestep subfolders
            # named like t0, t1, ... and map them to results[t] = {"path": str(t_folder)}
            try:
                subdirs = [d for d in sorted(p.iterdir()) if d.is_dir() and d.name.startswith("t") and d.name[1:].isdigit()]
                if subdirs:
                    for d in subdirs:
                        try:
                            t = int(d.name[1:])
                            results[t] = {"path": str(d)}
                        except Exception:
                            # ignore non-numeric t* dirs
                            pass
                    return results
            except Exception:
                # fall through to legacy manifest/zarr discovery
                pass
            manifest = p / "manifest.json"
            if not manifest.exists():
                # try to discover files by pattern embeddings_t*.zarr
                zarrs = sorted(p.glob("embeddings_t*.zarr"))
                if not zarrs:
                    raise FileNotFoundError(f"No manifest.json or embeddings_* found in {p}")
                for z in zarrs:
                    tstr = z.stem.split("_t")[-1]
                    import zarr as _zarr
                    f = _zarr.open(str(z), mode="r")
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"Skipping {z}: no array-like 'features' dataset found.")
                        except Exception:
                            pass
                        continue
                    # Try to surface input/original size metadata expected by downstream code
                    attrs = getattr(feats, "attrs", {}) or {}
                    # If the zarr store doesn't contain tiling metadata, synthesize
                    # a conservative `input_size`/`original_size` so downstream
                    # prompt-based segmentation treats this as a non-tiled embedding
                    # (avoids accessing attrs['shape'] which may be missing).
                    input_size = attrs.get("input_size")
                    original_size = attrs.get("original_size")
                    if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                        # fallback: infer spatial size from the last two dimensions
                        try:
                            inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                            input_size = input_size or inferred
                            original_size = original_size or inferred
                        except Exception:
                            input_size = input_size or None
                            original_size = original_size or None
                    results[int(tstr)] = {"features": feats, "input_size": input_size, "original_size": original_size}
                return results

            with open(manifest, "r") as fh:
                manifestd = json.load(fh)
            for tstr, fname in manifestd.get("files", {}).items():
                t = int(tstr)
                filep = p / fname
                if filep.suffix == ".npz":
                    data = np.load(str(filep))
                    # assume key 't{t}' exists
                    arr = data.get(f"t{t}")
                    # npz does not contain metadata about input/original size -> leave as None
                    results[t] = {"features": arr, "input_size": None, "original_size": None}
                else:
                    import zarr as _zarr
                    f = _zarr.open(str(filep), mode="r")
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"Skipping {filep}: no array-like 'features' dataset found.")
                        except Exception:
                            pass
                        continue
                    attrs = getattr(feats, "attrs", {}) or {}
                    input_size = attrs.get("input_size")
                    original_size = attrs.get("original_size")
                    if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                        try:
                            inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                            input_size = input_size or inferred
                            original_size = original_size or inferred
                        except Exception:
                            input_size = input_size or None
                            original_size = original_size or None
                    if lazy:
                        results[t] = {"features": feats, "input_size": input_size, "original_size": original_size}
                    else:
                        results[t] = {"features": feats[:], "input_size": input_size, "original_size": original_size}
            return results

        elif p.is_file() and p.suffix == ".npz":
            data = np.load(str(p))
            for key in data.files:
                if key.startswith("t"):
                    t = int(key[1:])
                    results[t] = {"features": data[key], "input_size": None, "original_size": None}
            return results
        else:
            raise FileNotFoundError(f"No embeddings found at {path}")

    def set_embeddings_folder(self, path: str | Path, lazy: bool = True):
        """Set a directory containing per-timestep embeddings and load the mapping.

        Stores the path in `self._last_embeddings_dir` and populates `self.embeddings_4d`.
        Returns the loaded mapping (t -> embedding entry).
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Embeddings path does not exist: {p}")
        self._last_embeddings_dir = str(p)
        loaded = self.load_saved_embeddings(p, lazy=lazy)
        # store mapping in per-timestep cache
        try:
            self.embeddings_4d = {int(k): v for k, v in loaded.items()}
        except Exception:
            self.embeddings_4d = loaded
        return self.embeddings_4d

    def reload_embeddings_from_last(self, lazy: bool = True):
        """Reload embeddings mapping from the last set embeddings directory.

        Raises RuntimeError if no directory was previously set via `set_embeddings_folder` or
        by the compute/save helpers. Returns the loaded mapping.
        """
        if not getattr(self, "_last_embeddings_dir", None):
            raise RuntimeError("No last embeddings directory is set. Call set_embeddings_folder(path) first.")
        return self.set_embeddings_folder(self._last_embeddings_dir, lazy=lazy)

    def _materialize_embedding_entry(self, entry):
        """Best-effort materialize a saved embedding entry.

        Accepts:
          - dict with 'features' -> returned as-is
          - dict with 'path' -> path to a .zarr store (or parent dir)
          - str path -> treated like dict with 'path'

        Returns embedding dict or None on failure.
        """
        if entry is None:
            return None
        # already materialized
        if isinstance(entry, dict) and "features" in entry:
            return entry

        p = None
        if isinstance(entry, dict) and "path" in entry and entry["path"]:
            p = Path(entry["path"])
        elif isinstance(entry, str):
            p = Path(entry)

        if p is None:
            return None

        # If p points to a per-timestep folder like 't3', treat it as a lazy
        # entry referencing that folder and return a simple {'path': str(p)}
        try:
            if p.is_dir() and p.name.startswith("t") and p.name[1:].isdigit():
                return {"path": str(p)}
        except Exception:
            pass

        # If given a parent directory, try to load via manifest (may return a mapping)
        try:
            if p.is_dir():
                loaded = self.load_saved_embeddings(p, lazy=True)
                # prefer the exact timestep if present; otherwise return the mapping
                return loaded
        except Exception:
            pass

        # If it's a zarr store, open it and return the 'features' object with synthesized metadata
        try:
            if p.suffix == ".zarr" or (p.exists() and p.is_dir() and any(p.glob("*.zarr"))):
                import zarr as _zarr

                # If p is a file-like zarr store, open it directly
                try:
                    f = _zarr.open(str(p), mode="r")
                except Exception:
                    # maybe the path points to a file-like directory; try opening parent
                    try:
                        f = _zarr.open(str(p), mode="r")
                    except Exception:
                        f = None

                if f is not None:
                    feats = _select_array_from_zarr_group(f)
                    if feats is None:
                        try:
                            show_info(f"No suitable array-like dataset found inside {p}; cannot materialize embeddings.")
                        except Exception:
                            pass
                        return None
                    attrs = getattr(feats, "attrs", {}) or {}
                    input_size = attrs.get("input_size")
                    original_size = attrs.get("original_size")
                    if input_size is None and ("shape" not in attrs and "tile_shape" not in attrs):
                        try:
                            inferred = (int(feats.shape[-2]), int(feats.shape[-1]))
                            input_size = input_size or inferred
                            original_size = original_size or inferred
                        except Exception:
                            input_size = input_size or None
                            original_size = original_size or None
                    return {"features": feats, "input_size": input_size, "original_size": original_size}
        except Exception:
            pass

        # Last resort: if it's an npz file, let load_saved_embeddings handle it
        try:
            if p.exists() and p.suffix == ".npz":
                loaded = self.load_saved_embeddings(p)
                # return the first timestep's entry if single
                if isinstance(loaded, dict):
                    # if only one entry, return that
                    if len(loaded) == 1:
                        return list(loaded.values())[0]
                    return loaded
        except Exception:
            pass

        return None

    def _ensure_embeddings_active_for_t(self, t: int):
        """If we have cached embeddings for timestep t, activate them on AnnotatorState.

        This will materialize lazy entries if necessary.
        """
        try:
            if not hasattr(self, "embeddings_4d") or self.embeddings_4d is None:
                return
            # Detach any previously active embeddings if switching timesteps
            try:
                state_detach = AnnotatorState()
                if getattr(self, "_active_embedding_t", None) is not None and self._active_embedding_t != int(t):
                    try:
                        state_detach.image_embeddings = None
                        state_detach.embedding_path = None
                    except Exception:
                        pass
                    try:
                        self._active_embedding_t = None
                    except Exception:
                        pass
            except Exception:
                pass

            entry = self.embeddings_4d.get(int(t))
            if entry is None:
                # No embeddings for this timestep: ensure global state is cleared
                try:
                    state_clear = AnnotatorState()
                    state_clear.image_embeddings = None
                    state_clear.embedding_path = None
                except Exception:
                    pass
                return

            # If entry is a mapping of multiple timesteps (returned from load_saved_embeddings on a dir),
            # prefer the exact t key.
            if isinstance(entry, dict) and any(isinstance(k, str) and k.isdigit() for k in entry.keys()):
                # already a mapping produced by load_saved_embeddings; pick the t entry if present
                maybe = entry.get(str(t)) or entry.get(int(t))
                if maybe is not None:
                    entry = maybe

            # If entry is a lazy path or string, materialize. Do this in background
            # to avoid blocking the UI for large zarr loads.
            if not (isinstance(entry, dict) and "features" in entry):
                # If already loading, return early
                if self._embedding_loading.get(int(t), False):
                    return

                # Mark as loading and spawn background thread
                self._embedding_loading[int(t)] = True
                try:
                    show_info(f"Materializing embeddings for timestep {t} in background...")
                except Exception:
                    pass

                def _bg():
                    try:
                        mat = self._materialize_embedding_entry(entry)
                        if mat is not None:
                            try:
                                self.embeddings_4d[int(t)] = mat
                            except Exception:
                                pass
                            # once materialized, call this method again to perform activation
                            try:
                                # clear loading flag before re-entering
                                self._embedding_loading[int(t)] = False
                            except Exception:
                                pass
                            try:
                                # call activation synchronously now that mat exists
                                self._ensure_embeddings_active_for_t(t)
                            except Exception:
                                pass
                            try:
                                show_info(f"Embeddings for timestep {t} are ready.")
                            except Exception:
                                pass
                        else:
                            try:
                                show_info(f"Failed to materialize embeddings for timestep {t}.")
                            except Exception:
                                pass
                    finally:
                        try:
                            self._embedding_loading[int(t)] = False
                        except Exception:
                            pass

                thread = threading.Thread(target=_bg, daemon=True)
                thread.start()
                return

            # Finally, set AnnotatorState's image_embeddings to this dict so downstream code uses it
            try:
                state = AnnotatorState()
                # if predictor is missing we'll (re)initialize it for this timestep
                image3d = None
                try:
                    image3d = self.image_4d[int(t)]
                except Exception:
                    image3d = None

                # If the entry is already materialized and contains features, use it as save_path
                save_path = None
                if isinstance(entry, dict) and "features" in entry:
                    save_path = entry
                elif isinstance(entry, dict) and entry.get("path"):
                    save_path = str(entry.get("path"))
                elif isinstance(entry, str):
                    save_path = entry

                # Determine a model_type to use if predictor is not present
                try:
                    model_type = getattr(state.predictor, "model_type", None) or getattr(state.predictor, "model_name", None)
                except Exception:
                    model_type = None
                if model_type is None:
                    try:
                        model_type = _vutil._DEFAULT_MODEL
                    except Exception:
                        model_type = "vit_b_lm"

                # Initialize predictor / embeddings for this timestep. If predictor already exists,
                # initialize_predictor will reuse it.
                try:
                    if image3d is not None:
                        state.initialize_predictor(
                            image3d,
                            model_type=model_type,
                            ndim=3,
                            save_path=save_path,
                            predictor=getattr(state, "predictor", None),
                            prefer_decoder=getattr(state, "decoder", None) is not None,
                        )
                except Exception:
                    # Continue even if predictor init fails; downstream code may handle it.
                    pass

                # ensure state.image_embeddings references the entry
                try:
                    state.image_embeddings = entry
                    try:
                        self._active_embedding_t = int(t)
                    except Exception:
                        pass
                except Exception:
                    pass

                # also set embedding_path if available (do this early so loaders can access files)
                try:
                    if isinstance(entry, dict) and entry.get("path"):
                        state.embedding_path = str(entry.get("path"))
                    elif isinstance(entry, str):
                        state.embedding_path = entry
                except Exception:
                    pass

                # Ensure image_shape/scale/name are set for downstream widgets
                try:
                    if image3d is not None:
                        state.image_shape = tuple(image3d.shape)
                except Exception:
                    pass
                try:
                    layer = self._viewer.layers.get("raw_4d", None)
                    if layer is not None:
                        scale = getattr(layer, "scale", None)
                        if scale is not None and len(scale) >= 4:
                            state.image_scale = tuple(scale[1:])
                        elif scale is not None and len(scale) == 3:
                            state.image_scale = tuple(scale)
                except Exception:
                    pass

                # Ensure AMG state exists. If embeddings were saved on disk (embedding_path), try to load
                # the precomputed amg/is state. Otherwise compute AMG state in-memory.
                try:
                    if state.amg_state is None:
                        # If embedding_path exists on disk, load cached AMG/IS state
                        if getattr(state, "embedding_path", None):
                            try:
                                if state.decoder is not None:
                                    state.amg_state = _load_is_state(state.embedding_path)
                                else:
                                    state.amg_state = _load_amg_state(state.embedding_path)
                            except Exception:
                                state.amg_state = None

                        # If still missing and we have in-memory embeddings, compute AMG state now
                        if state.amg_state is None and isinstance(state.image_embeddings, dict) and "features" in state.image_embeddings and image3d is not None:
                            try:
                                is_tiled = state.image_embeddings.get("input_size") is None
                                amg = instance_segmentation.get_amg(state.predictor, is_tiled=is_tiled, decoder=state.decoder)
                                # initialize amg on the full 3D volume
                                amg.initialize(image3d, image_embeddings=state.image_embeddings, verbose=False)
                                state.amg = amg
                                state.amg_state = amg.get_state()
                            except Exception:
                                # best-effort; leave amg_state None if computation fails
                                pass
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    def _on_dims_current_step(self, event):
        """Handle napari dims current_step changes (time slider).

        Persists current 3D edits and loads the new timestep into the 3D views.
        """
        # robustly read new step value
        try:
            val = getattr(event, "value", None)
            if val is None:
                val = getattr(event, "current_step", None) or getattr(self._viewer.dims, "current_step", None)
        except Exception:
            val = None
        if val is None:
            return
        try:
            new_t = int(val[0]) if isinstance(val, (list, tuple)) else int(val)
        except Exception:
            return

        # If the time index didn't actually change, do not treat this
        # as a timestep switch — the user may have adjusted Z/Y/X sliders.
        if new_t == getattr(self, "current_timestep", None):
            # nothing to do for time change; allow Napari to handle other axes
            return

        # persist point prompts for previous timestep
        try:
            prev = getattr(self, "current_timestep", None)
            if prev is not None and "point_prompts" in self._viewer.layers:
                pts = np.array(self._viewer.layers["point_prompts"].data)
                self.point_prompts_4d[prev] = pts if pts.size else None
        except Exception:
            pass

        # keep internal arrays referencing layer data (edits mutate layer.data)
        try:
            if "committed_objects_4d" in self._viewer.layers:
                self.segmentation_4d = self._viewer.layers["committed_objects_4d"].data
            if "current_object_4d" in self._viewer.layers:
                self.current_object_4d = self._viewer.layers["current_object_4d"].data
            if "auto_segmentation_4d" in self._viewer.layers:
                self.auto_segmentation_4d = self._viewer.layers["auto_segmentation_4d"].data
        except Exception:
            pass

        # switch displayed timestep without recreating layers
        try:
            if new_t != self.current_timestep:
                # Use the timestep embedding manager if available to lazily load
                # and activate per-timestep zarr embeddings. Falls back to the
                # existing activation method when no manager is present.
                try:
                    mgr = getattr(self, "timestep_embedding_manager", None)
                    if mgr is not None:
                        try:
                            mgr.on_timestep_changed(new_t)
                        except Exception:
                            # fallback
                            try:
                                self._ensure_embeddings_active_for_t(new_t)
                            except Exception:
                                pass
                    else:
                        try:
                            self._ensure_embeddings_active_for_t(new_t)
                        except Exception:
                            pass
                except Exception:
                    pass
                self._load_timestep(new_t)
        except Exception:
            pass

    def previous_timestep(self):
        """Move to the previous timestep (if available)."""
        self.save_current_object_to_4d()
        self.save_point_prompts()
        if self.current_timestep - 1 >= 0:
            self._load_timestep(self.current_timestep - 1)
        else:
            print("🚫 Already at first timestep.")

    def next_timestep(self):
        """Move to the next timestep (if available)."""
        self.save_current_object_to_4d()
        self.save_point_prompts()
        if self.current_timestep + 1 < self.n_timesteps:
            self._load_timestep(self.current_timestep + 1)
        else:
            print("🚫 Already at last timestep.")

    def _update_timestep_controls(self):
        """Placeholder for UI controls update (no-op)."""
        pass

    # ----------------- Automatic segmentation helpers for 4D -----------------
    def auto_segment_timestep(self, t: int, mode: str = "auto", device: str | None = None, tile_shape=None, halo=None, gap_closing: int | None = None, min_z_extent: int | None = None, with_background: bool = True, min_object_size: int = 100, prefer_decoder: bool = True):
        """Run automatic 3D segmentation for a single timestep and store result in auto_segmentation_4d[t].

        This uses the project's `automatic_3d_segmentation` implementation which handles tiled vs untiled
        embeddings and decoder vs AMG-based segmentation.
        """
        if self.image_4d is None:
            raise RuntimeError("No 4D image loaded")
        if not (0 <= t < self.n_timesteps):
            raise IndexError("t out of range")

        # Ensure embeddings and predictor are initialized for this timestep.
        state = AnnotatorState()
        # If no predictor or embeddings are present for this timestep, compute them.
        if state.predictor is None or state.image_embeddings is None:
            # compute embeddings for this timestep (store per-timestep only)
            self.compute_embeddings_for_timestep(t=t, model_type=None, device=device, save_path=None, tile_shape=tile_shape, halo=halo, prefer_decoder=prefer_decoder)
            # After computing, bind the per-timestep embeddings into the global state
            try:
                emb = self.embeddings_4d.get(int(t))
                if emb is None:
                    try:
                        show_info(f"❌ No embeddings available for timestep {t}; cannot run segmentation.")
                    except Exception:
                        print(f"❌ No embeddings available for timestep {t}; cannot run segmentation.")
                    return None
                state.image_embeddings = emb
                # set embedding path if available
                try:
                    if isinstance(emb, dict) and emb.get("path"):
                        state.embedding_path = str(emb.get("path"))
                except Exception:
                    pass
                # mark active
                try:
                    self._active_embedding_t = int(t)
                except Exception:
                    pass
            except Exception:
                pass

        predictor = state.predictor

        # Determine if tiled embeddings are used.
        is_tiled = False
        try:
            feats = state.image_embeddings.get("features") if isinstance(state.image_embeddings, dict) else None
            if feats is not None and hasattr(feats, "attrs") and feats.attrs.get("tile_shape") is not None:
                is_tiled = True
        except Exception:
            is_tiled = False

        # Create segmentor (AMG or decoder-based) using existing util.
        segmentor = instance_segmentation.get_amg(predictor, is_tiled=is_tiled, decoder=state.decoder)

        # Extract the 3D volume for this timestep
        vol3d = np.asarray(self.image_4d[int(t)])

        # Run automatic 3d segmentation
        seg = automatic_3d_segmentation(
            volume=vol3d,
            predictor=predictor,
            segmentor=segmentor,
            embedding_path=state.embedding_path,
            with_background=with_background,
            gap_closing=gap_closing,
            min_z_extent=min_z_extent,
            tile_shape=tile_shape,
            halo=halo,
            verbose=True,
            return_embeddings=False,
            min_object_size=min_object_size,
        )

        # Ensure segmentation output matches the volume shape (Z,Y,X). If not,
        # resize the segmentation volume (nearest-neighbour) to the target shape.
        try:
            target_shape = vol3d.shape
            if getattr(seg, "shape", None) != target_shape:
                seg = _sk_resize(seg.astype("float32"), target_shape, order=0, preserve_range=True, anti_aliasing=False).astype(seg.dtype)
        except Exception:
            # If resizing fails, continue and let assignment handle or raise.
            pass

        # Ensure our container exists
        if self.auto_segmentation_4d is None:
            self.auto_segmentation_4d = np.zeros_like(self.image_4d, dtype=np.uint32)
            try:
                if "auto_segmentation_4d" in self._viewer.layers:
                    self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
                else:
                    self._viewer.add_labels(data=self.auto_segmentation_4d, name="auto_segmentation_4d")
            except Exception:
                pass

        # Write back into the 4D auto segmentation container and refresh layer in-place
        try:
            self.auto_segmentation_4d[int(t)] = seg
            layer = self._viewer.layers.get("auto_segmentation_4d", None)
            if layer is not None:
                layer.data[int(t)] = seg
                try:
                    layer.refresh()
                except Exception:
                    pass
        except Exception:
            # As a fallback, try replacing the whole layer data
            try:
                if "auto_segmentation_4d" in self._viewer.layers:
                    self._viewer.layers["auto_segmentation_4d"].data = self.auto_segmentation_4d
            except Exception:
                pass

        print(f"✅ Auto-segmentation completed for timestep {t}")
        return seg

    def auto_segment_all_timesteps(self, mode: str = "auto", device: str | None = None, tile_shape=None, halo=None, gap_closing: int | None = None, min_z_extent: int | None = None, with_background: bool = True, min_object_size: int = 100, prefer_decoder: bool = True, overwrite: bool = False):
        """Run automatic 3D segmentation for every timestep and store results in auto_segmentation_4d.

        Returns a list of segmentation arrays per timestep.
        """
        results = []
        
        for t in range(self.n_timesteps):
            # skip existing unless overwrite
            if not overwrite and self.auto_segmentation_4d is not None:
                if np.any(self.auto_segmentation_4d[t]):
                    results.append(self.auto_segmentation_4d[t])
                    continue
            seg = self.auto_segment_timestep(t=t, mode=mode, device=device, tile_shape=tile_shape, halo=halo, gap_closing=gap_closing, min_z_extent=min_z_extent, with_background=with_background, min_object_size=min_object_size, prefer_decoder=prefer_decoder)
            results.append(seg)
        return results

    def remap_segment_id(self, timestep: int, old_id: int, new_id: int, propagate_forward: bool = False):
        """Remap a segment ID in a specific timestep.

        Args:
            timestep: The timestep containing the segment to remap
            old_id: The current ID of the segment
            new_id: The new ID to assign
            propagate_forward: If True, propagate the remapping to future timesteps
        """
        if self.segmentation_4d is None:
            raise ValueError("No segmentation available")
        if not (0 <= timestep < self.n_timesteps):
            raise ValueError(f"Invalid timestep {timestep}")

        # Do the remapping for the specified timestep
        mask = self.segmentation_4d[timestep] == old_id
        if not np.any(mask):
            print(f"⚠️ No object with ID {old_id} found in timestep {timestep}")
            return

        self.segmentation_4d[timestep][mask] = new_id

        # Update the view
        try:
            layer = self._viewer.layers.get("committed_objects_4d")
            if layer is not None:
                layer.data = self.segmentation_4d
                try:
                    layer.refresh()
                except Exception:
                    pass
        except Exception:
            pass

        # Propagate to future timesteps if requested
        if propagate_forward:
            for t in range(timestep + 1, self.n_timesteps):
                mask = self.segmentation_4d[t] == old_id
                if np.any(mask):
                    self.segmentation_4d[t][mask] = new_id

        print(f"✅ Remapped segment ID {old_id} to {new_id} in timestep {timestep}"
              f"{' and propagated forward' if propagate_forward else ''}")

        # Update cache if it exists
        if self._segmentation_cache is not None:
            try:
                if 0 <= timestep < len(self._segmentation_cache):
                    self._segmentation_cache[timestep] = self.segmentation_4d[timestep].copy()
                    if propagate_forward:
                        for t in range(timestep + 1, min(self.n_timesteps, len(self._segmentation_cache))):
                            self._segmentation_cache[t] = self.segmentation_4d[t].copy()
            except Exception:
                pass
