"""
Loader functions for NeuroPAL prompts in the 4D annotator.

This module provides functions to load point prompts with neuron names
into the annotator_4d.py viewer.
"""

import numpy as np
import json
from pathlib import Path


def load_neuropal_prompts(prompt_dir, timestep=0):
    """
    Load NeuroPAL-derived point prompts with neuron names.
    
    Parameters
    ----------
    prompt_dir : str
        Directory containing the prompt files
    timestep : int, optional
        Timestep to load (default: 0)
    
    Returns
    -------
    dict
        Dictionary with:
        - 'prompts': (N, 3) array of [z, y, x] coordinates
        - 'ids': (N,) array of point IDs
        - 'names': dict mapping {ID: neuron_name}
        - 'success': bool indicating if load was successful
    """
    prompt_path = Path(prompt_dir)
    
    try:
        # Load files
        prompts_file = prompt_path / f"point_prompts_{timestep}.npy"
        ids_file = prompt_path / f"point_ids_{timestep}.npy"
        names_file = prompt_path / "neuron_names.json"
        
        # Check if all files exist
        missing_files = []
        if not prompts_file.exists():
            missing_files.append(str(prompts_file))
        if not ids_file.exists():
            missing_files.append(str(ids_file))
        if not names_file.exists():
            missing_files.append(str(names_file))
        
        if missing_files:
            error_msg = f"Missing files:\n" + "\n".join(missing_files)
            return {
                'success': False,
                'error': error_msg,
                'prompts': None,
                'ids': None,
                'names': None
            }
        
        # Load data
        point_prompts = np.load(prompts_file)
        point_ids = np.load(ids_file)
        
        with open(names_file, 'r') as f:
            neuron_names = json.load(f)
        
        # Convert keys to integers if they're strings
        neuron_names = {int(k): v for k, v in neuron_names.items()}
        
        # Validate data
        if point_prompts.shape[0] != point_ids.shape[0]:
            return {
                'success': False,
                'error': f"Mismatch: {point_prompts.shape[0]} prompts but {point_ids.shape[0]} IDs",
                'prompts': None,
                'ids': None,
                'names': None
            }
        
        if point_prompts.shape[1] != 3:
            return {
                'success': False,
                'error': f"Invalid prompt shape: {point_prompts.shape}. Expected (N, 3)",
                'prompts': None,
                'ids': None,
                'names': None
            }
        
        # Check that all IDs have names
        missing_names = [pid for pid in point_ids if pid not in neuron_names]
        if missing_names:
            print(f"Warning: Missing names for IDs: {missing_names}")
        
        return {
            'success': True,
            'prompts': point_prompts,
            'ids': point_ids,
            'names': neuron_names,
            'num_prompts': len(point_prompts),
            'timestep': timestep
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Error loading prompts: {str(e)}",
            'prompts': None,
            'ids': None,
            'names': None
        }


def create_neuron_name_layer_data(prompts, ids, names):
    """
    Create data for napari Points layer showing neuron names.
    
    Parameters
    ----------
    prompts : ndarray
        (N, 3) array of [z, y, x] coordinates
    ids : ndarray
        (N,) array of point IDs
    names : dict
        Dictionary mapping {ID: neuron_name}
    
    Returns
    -------
    dict
        Dictionary with:
        - 'coordinates': (N, 3) array for Points layer
        - 'text': dict with text properties
        - 'properties': dict with neuron metadata
    """
    # Create text labels
    text_labels = [names.get(int(pid), f"ID_{pid}") for pid in ids]
    
    # Create properties for hover info
    properties = {
        'neuron_id': [names.get(int(pid), f"ID_{pid}") for pid in ids],
        'point_id': ids.tolist(),
        'z': prompts[:, 0].tolist(),
        'y': prompts[:, 1].tolist(),
        'x': prompts[:, 2].tolist(),
    }
    
    # Text parameters for napari
    text = {
        'string': text_labels,
        'size': 12,
        'color': 'yellow',
        'anchor': 'center',
        'translation': np.array([0, 0, 10])  # Slight offset in X to avoid overlap
    }
    
    return {
        'coordinates': prompts.copy(),
        'text': text,
        'properties': properties
    }


def validate_prompt_format(prompt_dir):
    """
    Validate that a directory contains valid NeuroPAL prompt files.
    
    Parameters
    ----------
    prompt_dir : str
        Directory to validate
    
    Returns
    -------
    dict
        Validation results with 'valid' bool and 'message' str
    """
    prompt_path = Path(prompt_dir)
    
    if not prompt_path.exists():
        return {
            'valid': False,
            'message': f"Directory does not exist: {prompt_dir}"
        }
    
    if not prompt_path.is_dir():
        return {
            'valid': False,
            'message': f"Not a directory: {prompt_dir}"
        }
    
    # Check for required files
    required_files = []
    optional_files = []
    
    # Look for any timestep files
    prompt_files = list(prompt_path.glob("point_prompts_*.npy"))
    id_files = list(prompt_path.glob("point_ids_*.npy"))
    names_file = prompt_path / "neuron_names.json"
    
    if not prompt_files:
        return {
            'valid': False,
            'message': "No point_prompts_*.npy files found"
        }
    
    if not id_files:
        return {
            'valid': False,
            'message': "No point_ids_*.npy files found"
        }
    
    if not names_file.exists():
        return {
            'valid': False,
            'message': "neuron_names.json file not found"
        }
    
    # Extract available timesteps
    timesteps = sorted([
        int(f.stem.split('_')[-1]) 
        for f in prompt_files
    ])
    
    return {
        'valid': True,
        'message': f"Valid NeuroPAL prompt directory with {len(timesteps)} timestep(s)",
        'timesteps': timesteps,
        'num_files': len(prompt_files)
    }


def main():
    """Test loading functionality"""
    
    # Test directory
    test_dir = "/Users/arnlois/Desktop/neuropal_prompts"
    
    print("=" * 60)
    print("Testing NeuroPAL Prompt Loader")
    print("=" * 60)
    
    # Validate directory
    validation = validate_prompt_format(test_dir)
    print(f"\nValidation: {validation['message']}")
    
    if not validation['valid']:
        print("❌ Invalid directory")
        return
    
    # Load prompts
    print(f"\nLoading prompts from timestep 0...")
    result = load_neuropal_prompts(test_dir, timestep=0)
    
    if result['success']:
        print(f"✅ Successfully loaded {result['num_prompts']} prompts")
        print(f"\nNeuron names:")
        for pid, name in sorted(result['names'].items()):
            coord = result['prompts'][pid - 1]  # IDs start from 1
            print(f"  ID {pid}: {name} at ({coord[2]:.1f}, {coord[1]:.1f}, {coord[0]:.1f})")
        
        # Test layer data creation
        layer_data = create_neuron_name_layer_data(
            result['prompts'],
            result['ids'],
            result['names']
        )
        print(f"\n✅ Created layer data with {len(layer_data['text']['string'])} labels")
        
    else:
        print(f"❌ Failed to load: {result['error']}")


if __name__ == "__main__":
    main()
