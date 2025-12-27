from pynwb import NWBHDF5IO
import numpy as np

# -------- CONFIG --------
nwb_path = "/Users/arnlois/000981/Hermaphrodites/sub-20220327-h4/sub-20220327-h4_ses-20220327_ophys.nwb"
output_npz = "calcium_segment_range.npz"

START = 100
END   = 102   # exclusive
# ------------------------

io = NWBHDF5IO(nwb_path, "r")
nwb = io.read()

img = nwb.acquisition["CalciumImageSeries"]

# ---- Extract only the selected range ----
movie = img.data[START:END]   # SAFE: lazy HDF5 slicing

print("Extracted movie shape:", movie.shape)

mean_image = movie.mean(axis=0)
max_image = movie.max(axis=0)

np.savez(
    output_npz,
    movie=movie,
    mean_image=mean_image,
    max_image=max_image,
    start_frame=START,
    end_frame=END
)

io.close()
print("Saved:", output_npz)
