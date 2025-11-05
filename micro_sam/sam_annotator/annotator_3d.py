from typing import Optional, Tuple, Union

import napari
import numpy as np

import torch

from .. import util
from . import _widgets as widgets
from ._state import AnnotatorState
from ._annotator import _AnnotatorBase
from .util import _initialize_parser, _sync_embedding_widget, _load_amg_state, _load_is_state


class Annotator3d(_AnnotatorBase):

    def get_layer_by_name(self, name):
        """Helper to find a layer by name safely."""
        for layer in self._viewer.layers:
            if layer.name == name:
                return layer
        return None

    def _get_widgets(self):
        """Register widgets including MoveSegmentWidget."""
        autosegment = widgets.AutoSegmentWidget(self._viewer, with_decoder=self._with_decoder, volumetric=True)
        segment_nd = widgets.SegmentNDWidget(self._viewer, tracking=False)
        return {
            "segment": widgets.segment_slice(),
            "segment_nd": segment_nd,
            "autosegment": autosegment,
            "commit": widgets.commit(),
            "clear": widgets.clear_volume(),
            "move_segment": widgets.MoveSegmentWidget(self),
        }

    def move_segment(self, segment_id_str, source="committed_objects"):
        """Move a segment from one layer (committed or auto) to the current object layer."""
        try:
            segment_id = int(segment_id_str)
        except ValueError:
            print("Invalid segment ID.")
            return
        # Resolve source and current layer names robustly. Users running the 4D
        # annotator often have layers named '<name>_4d' (e.g. 'committed_objects_4d').
        def resolve_layer(name_candidates):
            for n in name_candidates:
                layer = self.get_layer_by_name(n)
                if layer is not None:
                    return layer, n
            return None, None

        # Candidate names to try for the source and current layers.
        src_candidates = [source, f"{source}_4d"]
        # Also try some common alternative names often present in the UI
        if source == "committed_objects":
            src_candidates.extend(["committed_objects_4d", "committed_objects_3d"])  # fallback names

        source_layer, resolved_source_name = resolve_layer(src_candidates)
        # Determine preferred current object layer name variants
        cur_candidates = ["current_object", "current_object_4d", "current_object_3d"]
        current_layer, resolved_current_name = resolve_layer(cur_candidates)

        if source_layer is None:
            available = [layer.name for layer in self._viewer.layers]
            print(f"Source layer '{source}' not found. Available layers: {available}")
            return

        if current_layer is None:
            available = [layer.name for layer in self._viewer.layers]
            print(f"Target layer 'current_object' not found. Available layers: {available}")
            return

        if source_layer is None or current_layer is None:
            print("Layers not found.")
            return

        source_data = source_layer.data
        current_data = current_layer.data

        # Determine whether layers are 4D (T, Z, Y, X). If so, operate only on
        # the currently selected timepoint to avoid cross-timestep moves.
        try:
            dims_cs = tuple(self._viewer.dims.current_step)
        except Exception:
            dims_cs = None

        if source_data.ndim == 4 and current_data.ndim == 4 and dims_cs is not None and len(dims_cs) >= 1:
            # assume time is the first axis (T, Z, Y, X)
            t = int(dims_cs[0])
            # Use the annotator helpers which are 4D-aware and will write into the
            # correct timestep slice. This avoids replacing Napari's underlying
            # layer.data array and preserves references used elsewhere.
            try:
                src_slice = self.get_layer_data(source)
                cur_slice = self.get_layer_data("current_object")
            except Exception:
                # Fallback to direct layer access if helper fails
                src_slice = source_data[t]
                cur_slice = current_data[t]

            mask = src_slice == segment_id
            if not np.any(mask):
                print(f"⚠️ No voxels found for segment ID {segment_id} in '{source}' at t={t}.")
                return

            # Update slices in-place then write back using set_layer_data so the
            # _AnnotatorBase logic can handle 4D vs 3D appropriately.
            cur_slice = cur_slice.copy()
            src_slice = src_slice.copy()
            cur_slice[mask] = 1
            src_slice[mask] = 0

            try:
                # set_layer_data will write into the correct timestep for 4D layers
                self.set_layer_data(source, src_slice)
                self.set_layer_data("current_object", cur_slice)
            except Exception:
                # Last-resort fallback: replace full layer.data (preserve previous behavior)
                source_data[t] = src_slice
                current_data[t] = cur_slice
                source_layer.data = source_data
                current_layer.data = current_data

            print(f"✅ Moved segment {segment_id} from '{source}' to 'current_object' at t={t}.")

        else:
            # fallback: operate on the full array (3d or 2d)
            mask = source_data == segment_id
            if not np.any(mask):
                print(f"⚠️ No voxels found for segment ID {segment_id} in '{source}'.")
                return

            with source_layer.events.data.blocker(), current_layer.events.data.blocker():
                current_data[mask] = 1
                source_data[mask] = 0
                source_layer.data = source_data
                current_layer.data = current_data

            print(f"✅ Moved segment {segment_id} from '{source}' to 'current_object'.")

    def __init__(self, viewer: "napari.viewer.Viewer", reset_state: bool = True) -> None:
        self._with_decoder = AnnotatorState().decoder is not None
        super().__init__(viewer=viewer, ndim=3)

        # Set the expected annotator class to the state.
        state = AnnotatorState()

        # Reset the state.
        if reset_state:
            state.reset_state()

        state.annotator = self

    def _update_image(self, segmentation_result=None):
        super()._update_image(segmentation_result=segmentation_result)
        # Load the amg state from the embedding path.
        state = AnnotatorState()
        if self._with_decoder:
            state.amg_state = _load_is_state(state.embedding_path)
        else:
            state.amg_state = _load_amg_state(state.embedding_path)


def annotator_3d(
    image: np.ndarray,
    embedding_path: Optional[Union[str, util.ImageEmbeddings]] = None,
    segmentation_result: Optional[np.ndarray] = None,
    model_type: str = util._DEFAULT_MODEL,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    return_viewer: bool = False,
    viewer: Optional["napari.viewer.Viewer"] = None,
    precompute_amg_state: bool = False,
    checkpoint_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    prefer_decoder: bool = True,
) -> Optional["napari.viewer.Viewer"]:
    """Start the 3d annotation tool for a given image volume.

    Args:
        image: The volumetric image data.
        embedding_path: Filepath where to save the embeddings
            or the precompted image embeddings computed by `precompute_image_embeddings`.
        segmentation_result: An initial segmentation to load.
            This can be used to correct segmentations with Segment Anything or to save and load progress.
            The segmentation will be loaded as the 'committed_objects' layer.
        model_type: The Segment Anything model to use. For details on the available models check out
            https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models.
        tile_shape: Shape of tiles for tiled embedding prediction.
            If `None` then the whole image is passed to Segment Anything.
        halo: Shape of the overlap between tiles, which is needed to segment objects on tile borders.
        return_viewer: Whether to return the napari viewer to further modify it before starting the tool.
            By default, does not return the napari viewer.
        viewer: The viewer to which the Segment Anything functionality should be added.
            This enables using a pre-initialized viewer.
        precompute_amg_state: Whether to precompute the state for automatic mask generation.
            This will take more time when precomputing embeddings, but will then make
            automatic mask generation much faster. By default, set to 'False'.
        checkpoint_path: Path to a custom checkpoint from which to load the SAM model.
        device: The computational device to use for the SAM model.
            By default, automatically chooses the best available device.
        prefer_decoder: Whether to use decoder based instance segmentation if
            the model used has an additional decoder for instance segmentation.
            By default, set to 'True'.

    Returns:
        The napari viewer, only returned if `return_viewer=True`.
    """

    # Initialize the predictor state.
    state = AnnotatorState()
    state.image_shape = image.shape[:-1] if image.ndim == 4 else image.shape
    state.initialize_predictor(
        image, model_type=model_type, save_path=embedding_path,
        halo=halo, tile_shape=tile_shape, ndim=3, precompute_amg_state=precompute_amg_state,
        checkpoint_path=checkpoint_path, device=device, prefer_decoder=prefer_decoder,
        use_cli=True,
    )

    if viewer is None:
        viewer = napari.Viewer()

    viewer.add_image(image, name="image")
    annotator = Annotator3d(viewer, reset_state=False)

    # Trigger layer update of the annotator so that layers have the correct shape.
    # And initialize the 'committed_objects' with the segmentation result if it was given.
    annotator._update_image(segmentation_result=segmentation_result)

    # Add the annotator widget to the viewer and sync widgets.
    viewer.window.add_dock_widget(annotator)
    _sync_embedding_widget(
        widget=state.widgets["embeddings"],
        model_type=model_type if checkpoint_path is None else state.predictor.model_type,
        save_path=embedding_path,
        checkpoint_path=checkpoint_path,
        device=device,
        tile_shape=tile_shape,
        halo=halo
    )

    if return_viewer:
        return viewer

    napari.run()


def main():
    """@private"""
    parser = _initialize_parser(description="Run interactive segmentation for an image volume.")
    args = parser.parse_args()
    image = util.load_image_data(args.input, key=args.key)

    if args.segmentation_result is None:
        segmentation_result = None
    else:
        segmentation_result = util.load_image_data(args.segmentation_result, key=args.segmentation_key)

        annotator_3d(
            image, embedding_path=args.embedding_path,
            segmentation_result=segmentation_result,
            model_type=args.model_type, tile_shape=args.tile_shape, halo=args.halo,
            checkpoint_path=args.checkpoint, device=args.device,
            precompute_amg_state=args.precompute_amg_state, prefer_decoder=args.prefer_decoder,
        )
