import numpy as np
from pynwb import NWBHDF5IO

nwb_path = "/Users/arnlois/000981/Hermaphrodites/sub-20220327-h4/sub-20220327-h4_ses-20220327_ophys.nwb"

# -----------------------------
# Load NWB
# -----------------------------
io = NWBHDF5IO(nwb_path, mode="r")
nwb = io.read()

# -----------------------------
# Extract calcium movie
# -----------------------------
image_series = nwb.acquisition["CalciumImageSeries"]

image_data = np.asarray(image_series.data)
print("Raw shape:", image_data.shape)

# -----------------------------
# Ensure (T, Z, Y, X)
# -----------------------------
if image_data.ndim == 3:
    # (T, Y, X) → (T, 1, Y, X)
    image_data = image_data[:, None, :, :]

print("Final shape:", image_data.shape)

# -----------------------------
# Optional: timestamps
# -----------------------------
timestamps = (
    image_series.timestamps[:]
    if image_series.timestamps is not None
    else None
)

# -----------------------------
# Save NPZ
# -----------------------------
np.savez(
    "calcium_movie.npz",
    image_4d=image_data,
    timestamps=timestamps
)

io.close()