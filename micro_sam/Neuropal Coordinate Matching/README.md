# NeuroPAL Integration - Complete Workflow Guide

This guide explains how to use NeuroPAL neuron identifications in your 4D calcium imaging segmentation workflow.

## Overview

The NeuroPAL integration allows you to:
1. Extract neuron centroids from NWB NeuroPAL files
2. Export them as point prompts for the 4D annotator
3. Load point prompts with neuron names (AVAL, AVAR, etc.)
4. Segment neurons with automatic ID assignment
5. Extract fluorescence traces with real neuron names

---

## Files in this Directory

- **`extract_neuron_coordinates.py`** - Extract centroids from NWB NeuroPAL data
- **`export_neuropal_prompts.py`** - Convert centroids to point prompt format
- **`load_neuropal_prompts.py`** - Utility functions for loading prompts (used by annotator)

---

## Complete Workflow

### Step 1: Extract NeuroPAL Centroids

First, identify which neurons you want to segment. Use `extract_neuron_coordinates.py` to see available neurons:

```python
python extract_neuron_coordinates.py
```

This will list all neurons in your NWB file and show the coordinates for one example neuron.

### Step 2: Export Point Prompts

Edit `export_neuropal_prompts.py` and set:
- `nwb_path`: Path to your NWB file
- `output_dir`: Where to save the prompt files
- `neuron_list`: Which neurons to export

Example:
```python
nwb_path = "/path/to/your/nwb_file.nwb"
output_dir = "/Users/arnlois/Desktop/neuropal_prompts"
neuron_list = ['AVAL', 'AVAR', 'AWCL', 'AWCR', 'RIML', 'RIMR']
```

Then run:
```bash
python export_neuropal_prompts.py
```

This creates three files in the output directory:
- `point_prompts_0.npy` - Coordinates (N, 3) array with [z, y, x]
- `point_ids_0.npy` - Sequential IDs (1, 2, 3, ...)
- `neuron_names.json` - Mapping {1: 'AVAL', 2: 'AVAR', ...}

### Step 3: Load Prompts in 4D Annotator

1. Open your 4D annotator:
   ```bash
   python -m micro_sam.sam_annotator.annotator_4d
   ```

2. Load your calcium imaging data as usual

3. Generate embeddings (if not already done)

4. Click **"Load NeuroPAL Prompts"** button (purple button in Point Prompt Tools section)

5. Select the directory containing your prompt files (e.g., `/Users/arnlois/Desktop/neuropal_prompts`)

6. Point prompts will appear at the neuron locations with:
   - Different colors for each neuron ID
   - Neuron names displayed in a separate "Neuron_Names" layer

### Step 4: Segment Neurons

1. Navigate to timestep 0 (where prompts were loaded)

2. Click **"Commit Current Segmentation"** to segment all neurons with point prompts

3. Optional: Use **"Propagate Object Forward"** or **"Propagate Object Backward"** to track neurons across time

4. The segmentation will preserve neuron IDs (1, 2, 3...) that correspond to your neuron names

### Step 5: Save Segmentation

1. Click **"Save Segmentation"** button in the green "Save Segmentations" section

2. Choose where to save the NPZ file

3. This saves:
   - `image_4d`: Original calcium imaging data
   - `segmentation_4d`: Segmented neurons with IDs

4. **IMPORTANT**: Copy the `neuron_names.json` file to the same directory as your saved segmentation so that fluorescence extraction can use the neuron names

### Step 6: Extract Fluorescence Traces

Run the fluorescence extraction script:

```bash
python extract_fluorescence.py /path/to/your/segmentation.npz
```

The script will:
1. Automatically search for `neuron_names.json` in the same directory
2. Extract raw fluorescence for each neuron
3. Perform background subtraction
4. Calculate ΔF/F₀
5. Save CSV files with neuron names as column headers:
   - `traces_raw.csv` - Columns: Timepoint, AVAL, AVAR, AWCL, ...
   - `traces_corrected.csv` - Background-subtracted traces
   - `traces_dff.csv` - ΔF/F₀ traces
   - `traces_background.csv` - Global background trace
6. Generate plots with neuron names in titles

---

## File Format Details

### Point Prompts Format

**point_prompts_0.npy:**
```
Array shape: (N, 3)
Each row: [z, y, x] coordinate
Example:
[[15, 234, 456],   # First neuron
 [16, 235, 450],   # Second neuron
 [14, 240, 460]]   # Third neuron
```

**point_ids_0.npy:**
```
Array shape: (N,)
Sequential IDs starting from 1
Example: [1, 2, 3]
```

**neuron_names.json:**
```json
{
  "1": "AVAL",
  "2": "AVAR",
  "3": "AWCL"
}
```

### NPZ Segmentation Format

The saved segmentation contains:
- `image_4d`: (T, Z, Y, X) - Original calcium imaging data
- `segmentation_4d`: (T, Z, Y, X) - Segmented neurons
  - 0 = background
  - 1, 2, 3... = neuron IDs (corresponding to neuron_names.json)

---

## Tips & Troubleshooting

### Coordinate System Alignment

- NeuroPAL coordinates should match your calcium imaging coordinate system
- If neurons appear in the wrong locations, check:
  - Z-axis orientation (may need to flip)
  - Pixel spacing/resolution differences
  - Origin offsets

### Selecting Neurons to Export

**Option 1: Export specific neurons**
```python
neuron_list = ['AVAL', 'AVAR', 'AWCL', 'AWCR', 'RIML', 'RIMR']
summary = export_centroids_as_prompts(nwb_path, neuron_list, output_dir)
```

**Option 2: Export ALL neurons**
```python
summary = export_all_neurons(nwb_path, output_dir)
```

### Viewing NeuroPAL Data

Use the NeuroPAL loader to visualize the multichannel data:

```bash
python -m micro_sam.sam_annotator.neuropal_loader
```

This helps verify neuron locations before exporting prompts.

### Missing Neurons

If a neuron is not found in the NWB file:
1. Check the neuron ID spelling (case-sensitive)
2. List all available neurons: `python extract_neuron_coordinates.py`
3. The export script will skip missing neurons and continue with others

### Neuron Names in Fluorescence Traces

The `extract_fluorescence.py` script searches for `neuron_names.json` in:
1. Same directory as the NPZ file
2. Parent directory of the NPZ file
3. Parent directory / neuropal_prompts subfolder

Make sure to copy `neuron_names.json` to one of these locations!

---

## Example Workflow Commands

```bash
# 1. Export prompts from NeuroPAL data
cd "/Users/arnlois/micro-sam-4d/micro_sam/Neuropal Coordinate Matching"
python export_neuropal_prompts.py

# 2. Open 4D annotator and load prompts
cd /Users/arnlois/micro-sam-4d
python -m micro_sam.sam_annotator.annotator_4d
# Click "Load NeuroPAL Prompts" and select output directory

# 3. Segment, save, and copy neuron_names.json
cp /Users/arnlois/Desktop/neuropal_prompts/neuron_names.json /path/to/saved/segmentation/

# 4. Extract fluorescence traces
python -m micro_sam.sam_annotator.extract_fluorescence /path/to/saved/segmentation/my_segmentation.npz
```

---

## Color-Coded Point Prompts

Each neuron gets a unique ID and color in the annotator:
- ID 1 → Red
- ID 2 → Green  
- ID 3 → Blue
- ID 4 → Yellow
- etc.

The "Neuron_Names" layer shows text labels at each neuron location so you can verify which neuron is which.

---

## Advanced: Batch Processing

To process multiple NWB files:

```python
import os
from pathlib import Path
from export_neuropal_prompts import export_centroids_as_prompts

nwb_files = [
    "/path/to/subject1.nwb",
    "/path/to/subject2.nwb",
    "/path/to/subject3.nwb",
]

neuron_list = ['AVAL', 'AVAR', 'AWCL', 'AWCR']

for nwb_path in nwb_files:
    output_dir = Path(nwb_path).parent / "neuropal_prompts"
    print(f"\nProcessing: {nwb_path}")
    export_centroids_as_prompts(nwb_path, neuron_list, str(output_dir))
```

---

## Summary

The NeuroPAL integration provides a complete pipeline from neuron identification to fluorescence trace extraction:

**NWB NeuroPAL Data** → **Extract Centroids** → **Export as Point Prompts** → **Load in 4D Annotator** → **Segment with IDs** → **Extract Traces with Real Neuron Names**

All neuron names (AVAL, AVAR, etc.) are preserved throughout the entire workflow, appearing in:
- Point prompt labels in the viewer
- CSV column headers
- Plot titles and legends
- Statistical summaries

---

For questions or issues, check:
- Coordinate system alignment
- File paths in configuration
- neuron_names.json location for fluorescence extraction
