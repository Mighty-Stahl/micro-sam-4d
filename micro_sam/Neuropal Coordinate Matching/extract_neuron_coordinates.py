#!/usr/bin/env python3
"""
Extract the exact coordinates of a specific neuron from NeuroPAL data in an NWB file.

This script extracts 3D voxel coordinates for identified neurons from NeuroPAL imaging data
stored in NWB files. Each neuron has a single coordinate representing its centroid position.

Usage:
    python extract_neuron_coordinates.py

To extract coordinates for a different neuron, modify the `neuron_id` variable in the main() function.
To use a different NWB file, modify the `nwb_path` variable.

Available neuron IDs include: AVAL, AVAR, AWCL, AWCR, RIML, RIMR, and many others.
Run the script to see the complete list of available neurons in your file.
"""

from pynwb import NWBHDF5IO
import numpy as np
import sys


def extract_neuron_coordinates(nwb_path, neuron_id):
    """
    Extract coordinates for a specific neuron from NeuroPAL data.
    
    Parameters
    ----------
    nwb_path : str
        Path to the NWB file
    neuron_id : str
        Neuron ID to search for (e.g., 'AVAL', 'AVAR', 'AWCL', etc.)
    
    Returns
    -------
    dict
        Dictionary containing neuron information including coordinates
    """
    io = NWBHDF5IO(nwb_path, 'r')
    nwbfile = io.read()
    
    try:
        # Access NeuroPAL data
        neuropal_neurons = nwbfile.processing["NeuroPAL"]["NeuroPALSegmentation"]["NeuroPALNeurons"]
        
        # Get voxel masks (coordinates) and neuron labels
        voxel_mask = neuropal_neurons.voxel_mask[:]
        id_labels = neuropal_neurons.ID_labels[:]
        
        # Find the index of the requested neuron
        neuron_index = None
        for idx, label in enumerate(id_labels):
            if str(label) == neuron_id:
                neuron_index = idx
                break
        
        if neuron_index is None:
            io.close()
            return None
        
        # Extract coordinates for this neuron
        neuron_coords = voxel_mask[neuron_index]
        
        # The voxel_mask is typically a single tuple (x, y, z, value) representing centroid
        # Check if it's a tuple or structured array element
        if hasattr(neuron_coords, '__len__') and len(neuron_coords) >= 3:
            # Single coordinate (centroid) - could be tuple or array-like
            coord_list = [float(neuron_coords[i]) for i in range(min(len(neuron_coords), 4))]
            result = {
                'neuron_id': neuron_id,
                'index': neuron_index,
                'coordinate': {
                    'x': coord_list[0],
                    'y': coord_list[1],
                    'z': coord_list[2]
                }
            }
            if len(coord_list) > 3:
                result['value'] = coord_list[3]
        else:
            # Array of coordinates
            coords_array = np.array(neuron_coords)
            
            # Calculate centroid and bounding box
            if len(coords_array.shape) == 2 and coords_array.shape[0] > 0:
                # Extract x, y, z columns (first 3 columns)
                xyz_coords = coords_array[:, :3]
                
                centroid = np.mean(xyz_coords, axis=0)
                min_coords = np.min(xyz_coords, axis=0)
                max_coords = np.max(xyz_coords, axis=0)
                
                result = {
                    'neuron_id': neuron_id,
                    'index': neuron_index,
                    'num_voxels': len(coords_array),
                    'all_coordinates': xyz_coords.tolist(),
                    'centroid': {
                        'x': float(centroid[0]),
                        'y': float(centroid[1]),
                        'z': float(centroid[2])
                    },
                    'bounding_box': {
                        'min': {'x': float(min_coords[0]), 'y': float(min_coords[1]), 'z': float(min_coords[2])},
                        'max': {'x': float(max_coords[0]), 'y': float(max_coords[1]), 'z': float(max_coords[2])}
                    }
                }
            else:
                result = {
                    'neuron_id': neuron_id,
                    'index': neuron_index,
                    'raw_data': coords_array.tolist()
                }
        
        io.close()
        return result
        
    except Exception as e:
        io.close()
        raise Exception(f"Error extracting neuron coordinates: {e}")


def list_all_neurons(nwb_path):
    """
    List all available neurons in the NeuroPAL data.
    
    Parameters
    ----------
    nwb_path : str
        Path to the NWB file
    
    Returns
    -------
    list
        List of all neuron IDs
    """
    io = NWBHDF5IO(nwb_path, 'r')
    nwbfile = io.read()
    
    try:
        neuropal_neurons = nwbfile.processing["NeuroPAL"]["NeuroPALSegmentation"]["NeuroPALNeurons"]
        id_labels = neuropal_neurons.ID_labels[:]
        io.close()
        return [str(label) for label in id_labels]
    except Exception as e:
        io.close()
        raise Exception(f"Error listing neurons: {e}")


def main():
    # Example usage
    nwb_path = "/Users/arnlois/000981/Males/sub-20220329-m7/sub-20220329-m7_ses-20220329_ophys.nwb"
    
    # List all available neurons
    print("=" * 60)
    print("Available neurons in NeuroPAL data:")
    print("=" * 60)
    try:
        all_neurons = list_all_neurons(nwb_path)
        print(f"Total neurons found: {len(all_neurons)}")
        print("Neuron IDs:", ", ".join(sorted(all_neurons)))
        print()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Extract coordinates for a specific neuron
    neuron_id = "AVAL"  # CHANGE THIS to the neuron you want
    
    print("=" * 60)
    print(f"Extracting coordinates for neuron: {neuron_id}")
    print("=" * 60)
    
    try:
        result = extract_neuron_coordinates(nwb_path, neuron_id)
        
        if result is None:
            print(f"Neuron '{neuron_id}' not found in the file.")
        else:
            print(f"\nNeuron ID: {result['neuron_id']}")
            print(f"Index: {result['index']}")
            
            if 'coordinate' in result:
                print(f"\nCoordinate (voxel position):")
                print(f"  X: {result['coordinate']['x']:.2f}")
                print(f"  Y: {result['coordinate']['y']:.2f}")
                print(f"  Z: {result['coordinate']['z']:.2f}")
                if 'value' in result:
                    print(f"  Value: {result['value']:.2f}")
            elif 'num_voxels' in result:
                print(f"Number of voxels: {result['num_voxels']}")
                print(f"\nCentroid coordinates:")
                print(f"  X: {result['centroid']['x']:.2f}")
                print(f"  Y: {result['centroid']['y']:.2f}")
                print(f"  Z: {result['centroid']['z']:.2f}")
                
                print(f"\nBounding box:")
                print(f"  Min: X={result['bounding_box']['min']['x']:.2f}, "
                      f"Y={result['bounding_box']['min']['y']:.2f}, "
                      f"Z={result['bounding_box']['min']['z']:.2f}")
                print(f"  Max: X={result['bounding_box']['max']['x']:.2f}, "
                      f"Y={result['bounding_box']['max']['y']:.2f}, "
                      f"Z={result['bounding_box']['max']['z']:.2f}")
                
                print(f"\nFirst 5 voxel coordinates:")
                for i, coord in enumerate(result['all_coordinates'][:5]):
                    print(f"  Voxel {i+1}: X={coord[0]:.2f}, Y={coord[1]:.2f}, Z={coord[2]:.2f}")
                
                if result['num_voxels'] > 5:
                    print(f"  ... and {result['num_voxels'] - 5} more voxels")
            else:
                print(f"\nRaw data: {result.get('raw_data', 'N/A')}")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
