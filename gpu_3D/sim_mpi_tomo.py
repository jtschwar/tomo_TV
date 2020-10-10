# General 3D - SIRT Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
from pytvlib import *
import mpi_astra_ctvlib
import numpy as np
import time
from tqdm import tqdm
check_cuda()
########################################

vol_size = '256'
file_name = 'au_sto.h5'

# Algorithm
alg = 'SIRT'
initAlg = '' # Algorithm Parameters (ie Projection Order or Filter)

# Number of Iterations (Main Loop)
Niter = 100

# Descent Parameter and Reduction
beta0 = 0.5
beta_red = 0.995

# Save Final Reconstruction. 
saveRecon = True

##########################################

#Read Image. 
(fName, tiltAngles, original_volume) = load_h5_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape
Nproj = tiltAngles.shape[0]

# Initialize C++ Object.. 
tomo_obj = mpi_astra_ctvlib.mpi_astra_ctvlib(Nslice, Nray, Nproj, np.deg2rad(tiltAngles))
tomo_obj.initilizeInitialVolume()

# Load Volume and Collect Projections. 
for s in range(tomo_obj.NsliceLoc()):
    tomo_obj.setOriginalVolume(original_volume[s+tomo_obj.firstSlice(),:,:], s)

initialize_algorithm(tomo_obj, alg, initAlg)
tomo_obj.create_projections()

if tomo_obj.rank() == 0: print('Starting Reconstruction')
rmse_vec = np.zeros(Niter)

#Main Loop
for i in tqdm(range(Niter)): 

    run(tomo_obj, alg)

    rmse_vec[i] = tomo_obj.rmse()


if tomo_obj.rank() == 0: 
    print('RMSE: ', rmse_vec)
    print('Reconstruction Complete, Saving Data..')
    print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_' + alg + '_MPI'
meta = {'vol_size':vol_size,'Niter':Niter,'initAlg':initAlg}
results = {'rmse':rmse_vec}
mpi_save_results([fDir, fName], tomo_obj, saveRecon, meta, results)

