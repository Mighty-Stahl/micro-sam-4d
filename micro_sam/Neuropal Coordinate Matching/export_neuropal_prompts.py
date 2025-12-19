"""
Export NeuroPAL neuron centroids as point prompts for the 4D annotator.

This script converts neuron centroids from NWB files into the format used by the
annotator_4d.py point prompt system.

Output files:
- point_prompts_0.npy: (N, 3) array with [z, y, x] coordinates
- point_ids_0.npy: (N,) array with sequential IDs [1, 2, 3, ...]
- neuron_names.json: {1: 'AVAL', 2: 'AVAR', 3: 'AWCL', ...}
"""

import numpy as np
import json
from pathlib import Path
from extract_neuron_coordinates import extract_neuron_coordinates, list_all_neurons


def export_centroids_as_prompts(nwb_path, neuron_list, output_dir, timestep=0):
    """
    Export neuron centroids as point prompts for the 4D annotator.
    
    Parameters
    ----------
    nwb_path : str
        Path to the NWB file containing NeuroPAL data
    neuron_list : list of str
        List of neuron IDs to export (e.g., ['AVAL', 'AVAR', 'AWCL'])
    output_dir : str
        Directory to save the output files
    timestep : int, optional
        Timestep index for the point prompts (default: 0)
    
    Returns
    -------
    dict
        Summary of exported prompts with neuron names and coordinates
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Storage for all extracted centroids
    centroids = []
    neuron_names_map = {}
    failed_neurons = []
    
    print(f"\nExtracting centroids for {len(neuron_list)} neurons...")
    print("=" * 60)
    
    for neuron_id in neuron_list:
        try:
            result = extract_neuron_coordinates(nwb_path, neuron_id)
            
            if result is None:
                print(f"⚠️  Neuron '{neuron_id}' not found in the file")
                failed_neurons.append(neuron_id)
                continue
            
            # Extract centroid coordinates
            if 'centroid' in result:
                # Multi-voxel neuron
                x = result['centroid']['x']
                y = result['centroid']['y']
                z = result['centroid']['z']
            elif 'coordinate' in result:
                # Single-voxel neuron
                x = result['coordinate']['x']
                y = result['coordinate']['y']
                z = result['coordinate']['z']
            else:
                print(f"⚠️  Neuron '{neuron_id}' has unexpected coordinate format")
                failed_neurons.append(neuron_id)
                continue
            
            centroids.append({
                'neuron_id': neuron_id,
                'x': x,
                'y': y,
                'z': z
            })
            print(f"✓ {neuron_id}: ({x:.1f}, {y:.1f}, {z:.1f})")
            
        except Exception as e:
            print(f"⚠️  Error extracting '{neuron_id}': {e}")
            failed_neurons.append(neuron_id)
    
    if not centroids:
        print("\n❌ No valid centroids extracted!")
        return None
    
    print(f"\n✓ Successfully extracted {len(centroids)} centroids")
    if failed_neurons:
        print(f"⚠️  Failed to extract: {', '.join(failed_neurons)}")
    
    # Convert to annotator format
    # Point prompts: (N, 3) array with [z, y, x] coordinates
    point_prompts = np.array([[c['z'], c['y'], c['x']] for c in centroids])
    
    # Point IDs: (N,) array with sequential IDs starting from 1
    point_ids = np.arange(1, len(centroids) + 1)
    
    # Neuron names mapping: {ID: neuron_name}
    neuron_names_map = {int(pid): c['neuron_id'] for pid, c in zip(point_ids, centroids)}
    
    # Save files
    prompts_file = output_path / f"point_prompts_{timestep}.npy"
    ids_file = output_path / f"point_ids_{timestep}.npy"
    names_file = output_path / "neuron_names.json"
    
    np.save(prompts_file, point_prompts)
    np.save(ids_file, point_ids)
    
    with open(names_file, 'w') as f:
        json.dump(neuron_names_map, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Files saved:")
    print(f"  📄 {prompts_file}")
    print(f"  📄 {ids_file}")
    print(f"  📄 {names_file}")
    print("=" * 60)
    
    # Print summary
    summary = {
        'num_neurons': len(centroids),
        'neurons': neuron_names_map,
        'coordinates': {name: (c['x'], c['y'], c['z']) for c in centroids for name in [c['neuron_id']]},
        'output_dir': str(output_path),
        'failed': failed_neurons
    }
    
    print("\n📊 Summary:")
    print(f"  Exported: {len(centroids)} neurons")
    print(f"  Point IDs: 1 to {len(centroids)}")
    print(f"  Timestep: {timestep}")
    
    return summary


def export_all_neurons(nwb_path, output_dir, timestep=0):
    """
    Export ALL neurons from the NWB file as point prompts.
    
    Parameters
    ----------
    nwb_path : str
        Path to the NWB file containing NeuroPAL data
    output_dir : str
        Directory to save the output files
    timestep : int, optional
        Timestep index for the point prompts (default: 0)
    
    Returns
    -------
    dict
        Summary of exported prompts
    """
    print("Listing all available neurons...")
    all_neurons = list_all_neurons(nwb_path)
    print(f"Found {len(all_neurons)} neurons in the file")
    
    return export_centroids_as_prompts(nwb_path, all_neurons, output_dir, timestep)


def main():
    """Example usage"""
    
    # ===== CONFIGURATION =====
    nwb_path = "/Users/arnlois/000981/Males/sub-20220329-m7/sub-20220329-m7_ses-20220329_ophys.nwb"
    output_dir = "/Users/arnlois/Desktop/neuropal_prompts"
    timestep = 0  # First timestep (t=0)
    
    # Option 1: Export specific neurons
    neuron_list = [
        'AVAL', 'AVAR',  # Command interneurons
        'AWCL', 'AWCR',  # Chemosensory neurons
        'RIML', 'RIMR',  # Motor interneurons
        'ASEL', 'ASER',  # Sensory neurons
    ]
    
    print("=" * 60)
    print("NeuroPAL Prompt Exporter")
    print("=" * 60)
    print(f"NWB file: {nwb_path}")
    print(f"Output directory: {output_dir}")
    print(f"Timestep: {timestep}")
    print()
    
    # Export specific neurons
    summary = export_centroids_as_prompts(nwb_path, neuron_list, output_dir, timestep)
    
    # Option 2: Export ALL neurons (uncomment to use)
    # summary = export_all_neurons(nwb_path, output_dir, timestep)
    
    if summary:
        print("\n✅ Export completed successfully!")
        print(f"\n📌 Next steps:")
        print(f"  1. Open your 4D annotator")
        print(f"  2. Click 'Load NeuroPAL Prompts' button")
        print(f"  3. Select the directory: {output_dir}")
        print(f"  4. Point prompts will appear with neuron names")
        print(f"  5. Use 'Commit Current Segmentation' to segment neurons")


if __name__ == "__main__":
    main()
