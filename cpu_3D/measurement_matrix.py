# General 3D - ASD/TV Reconstruction with Positivity Constraint. 
# Intended for simulated datasets to measure RMSE and Volume's Original TV. 
# and to reconstruct large volume sizes (>1000^3) with Distributed Memory (OpenMPI)
# 
# This is to generate measurement matrix
# 

import sys, os
sys.path.append('./Utils')
from pytvlib import parallelRay, timer, load_data
#from mpi4py import MPI
import numpy as np
import h5py
import time
########################################
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-a", "--tiltAngles", default="Tilt_Series/au_sto_tiltAngles.npy", type=str)
parser.add_argument("-s", "--tiltSeries", default='Tilt_Series/256_au_sto_tiltser.npy', type=str)
parser.add_argument("-o", "--output", default='Measurement_Matrices/output.npy', type=str)
parser.add_argument("--hdf5", default=None, type=str)
args = parser.parse_args()
args.tiltSeries = args.tiltSeries.split('/')[1]
#args.tiltAngles = args.tiltAngles.split('/')[1]
vol_size = args.tiltSeries.split("_")[0] + "_"
file_name = args.tiltSeries[len(vol_size):]
(file_name, original_volume) = load_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape

# Generate Tilt Angles.
tiltAngles = np.load(args.tiltAngles, allow_pickle=True)
Nproj = tiltAngles.shape[0]
print("\n **** Generating Measurement Matrix ****")
print("           Data: ", args.tiltSeries)
print("    Volume size: ", vol_size[:-1])
print("       # slices: ", Nslice)
print("         # rays: ", Nray)
print("  # projections: ", Nproj)
print("-------------------------")

# Initialize C++ Object.. 
# Generate measurement matrix
t0 = time.time()
A = parallelRay(Nray, tiltAngles)
t1 = time.time()
print("* Total time spent: %s seconds" %(t1 - t0))
np.save(args.output, A)

if (args.hdf5==None):
    sub = len(args.output.split('.')[-1])
    args.hdf5 = args.output[:-sub]+"h5"
x, y = A.shape
#print("Size of A %s GB" %(x*y*sizeof(float)/1024/1024/1024))
h5=h5py.File(args.hdf5, 'w')
dset = h5.create_dataset("matrix", A.shape, dtype=np.float32, data=A)
dset.attrs["Nslice"] = Nslice
dset.attrs["Nray"] = Nray
dset.attrs["Vol"] = int(vol_size[:-1])
dset.attrs["Nproj"] = Nproj
dset.attrs["system"] = file_name
h5.close()
print("* Result is saved to %s and %s"%(args.output, args.hdf5))
