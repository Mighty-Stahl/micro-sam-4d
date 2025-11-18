
from __future__ import annotations

import argparse
import sys
import numpy as np

try:
    import napari
except Exception:  # pragma: no cover - runtime environment dependent
    napari = None

from micro_sam.sam_annotator.annotator_4d import MicroSAM4DAnnotator




# fake data
def main(argv=None):
    parser = argparse.ArgumentParser(description="Run MicroSAM 4D annotator demo")
    parser.add_argument("--fake-data", action="store_true", help="Use synthetic 4D data")
    parser.add_argument("--path", type=str, default=None, help="Path to .npz file containing a 4D array (T,Z,Y,X)")
    parser.add_argument("--T", type=int, default=6, help="timesteps")
    parser.add_argument("--Z", type=int, default=16, help="z slices")
    parser.add_argument("--Y", type=int, default=128, help="height")
    parser.add_argument("--X", type=int, default=128, help="width")
    parser.add_argument("--seed", type=int, default=0, help="random seed for fake data")
    args = parser.parse_args(argv)
    if napari is None:
        print("napari is required to run this demo. Install it with `pip install napari`")
        return 1

    image4d = None
    # If a path is provided, try to load the NPZ and auto-detect the array
    if args.path is not None:
        path = args.path
        print(f"Loading NPZ from: {path}")
        try:
            # Try memmap mode for large files; np.load on .npz may accept mmap_mode
            npz = np.load(path, mmap_mode='r')
        except TypeError:
            # older numpy versions may not accept mmap_mode for npz; fall back
            npz = np.load(path)
        except Exception as e:
            print(f"Failed to open NPZ: {e}")
            return 2

        # If the NPZ contains multiple arrays, pick a sensible name
        if isinstance(npz, np.lib.npyio.NpzFile):
            keys = list(npz.files)
            print(f"NPZ keys found: {keys}")
            chosen_key = None
            for candidate in ("data", "calcium"):
                if candidate in keys:
                    chosen_key = candidate
                    break
            if chosen_key is None and keys:
                chosen_key = keys[0]
            if chosen_key is None:
                print("No arrays found inside the NPZ file.")
                return 2
            try:
                arr = npz[chosen_key]
            except Exception as e:
                print(f"Failed to read array '{chosen_key}' from NPZ: {e}")
                return 2
            print(f"Using array '{chosen_key}' from NPZ")
            image4d = arr
        else:
            # single .npy file loaded
            image4d = npz

    default_npz_path = "/Users/arnlois/data/code/cropped2_sub-20190928-13_ses-20190928_ophys_calcium.npz"

    if image4d is None:
        print(f"Loading default NPZ from: {default_npz_path}")
        try:
            npz = np.load(default_npz_path, mmap_mode='r')
            if isinstance(npz, np.lib.npyio.NpzFile):
                keys = list(npz.files)
                print(f"NPZ keys found: {keys}")
                key = "data" if "data" in keys else keys[0]
                image4d = npz[key]
            else:
                image4d = npz
        except Exception as e:
            print(f"Failed to load default NPZ file: {e}")
            return 2

    # validate shape and dtype, and make sure it's a numpy array-like
    try:
        shape = getattr(image4d, 'shape', None)
        dtype = getattr(image4d, 'dtype', None)
        print(f"Loaded array shape={shape}, dtype={dtype}")
        if shape is None or len(shape) != 4:
            print("Loaded array is not 4D (T,Z,Y,X). Aborting.")
            return 2
    except Exception as e:
        print(f"Failed to inspect loaded array: {e}")
        return 2

    # Limit timestep sbased on what u like
    image4d = image4d[1:4]
    print(f"Trimmed: new shape = {image4d.shape}")

    # create Napari viewer and annotator
    viewer = napari.Viewer()
    annot = MicroSAM4DAnnotator(viewer)
    # Show the annotator dock so embedding/prompt widgets are available
    try:
        viewer.window.add_dock_widget(annot, area="right")
    except Exception:
        # older napari versions or non-GUI contexts may not support add_dock_widget
        pass

    # If the loaded array is an mmap-like array, pass as-is so Napari can
    # avoid copying large data where possible. The annotator will create
    # persistent 4D layers referencing this array.
    annot.update_image(image4d)

    print("viewer loaded")

    # show Napari GUI (blocking)
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

print("segmentation successful")