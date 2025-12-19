#!/usr/bin/env python3
"""
Demo: Extract coordinates for multiple neurons.
"""

from extract_neuron_coordinates import extract_neuron_coordinates

nwb_path = "/Users/arnlois/000981/Males/sub-20220329-m7/sub-20220329-m7_ses-20220329_ophys.nwb"

# Test with several neurons
test_neurons = ["AVAL", "AVAR", "AWCL", "AWCR", "RIML", "RIMR"]

print("Extracting coordinates for multiple neurons:")
print("=" * 60)

for neuron_id in test_neurons:
    result = extract_neuron_coordinates(nwb_path, neuron_id)
    if result:
        coord = result['coordinate']
        print(f"\n{neuron_id}: X={coord['x']:.1f}, Y={coord['y']:.1f}, Z={coord['z']:.1f}")
    else:
        print(f"\n{neuron_id}: NOT FOUND")
