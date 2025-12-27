from pynwb import NWBHDF5IO

# io = NWBHDF5IO("/Users/arnlois/data/code/sub-20190928-13_ses-20190928_ophys_calcium.npz", "r")
io = NWBHDF5IO("/Users/arnlois/000981/Hermaphrodites/sub-20220327-h4/sub-20220327-h4_ses-20220327_ophys.nwb", "r")
nwb = io.read()

print("Processing modules:")
print(nwb.processing.keys())

print("\nAcquisition:")
print(nwb.acquisition.keys())

io.close()


#to find out where calcium traces are storeed (keys)