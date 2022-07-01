# General 3D Reconstruction with Positivity Constraint. 

import mpi_astra_ctvlib
from tqdm import tqdm
from Utils.pytvlib import *
import numpy as np
check_cuda()
########################################

vol_size = '256'
file_name = 'au_sto.h5'

# Algorithm
alg = 'SIRT'
initAlg = '' # Algorithm Parameters (ie Projection Order or Filter)

# Number of Iterations (Main Loop)
Niter = 100

# Descent Parameter and Reduction (ART)
beta0, beta_red = 0.5, 0.995

# Save Final Reconstruction. 
saveRecon = True

##########################################

#Read Image. 
(fName, tiltAngles, original_volume) = load_h5_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape
Nproj = tiltAngles.shape[0]

# Initialize C++ Object.. 
tomo = mpi_astra_ctvlib.mpi_astra_ctvlib(Nslice, Nray, np.deg2rad(tiltAngles))
tomo.initialize_initial_volume()

# Load Volume and Collect Projections. 
for s in range(tomo.NsliceLoc()):
    tomo.set_original_volume(original_volume[s+tomo.firstSlice(),:,:], s)

initialize_algorithm(tomo, alg, initAlg)
tomo.create_projections()

if tomo.rank() == 0: print('Starting Reconstruction')
rmse_vec = np.zeros(Niter)

#Main Loop
for i in tqdm(range(Niter)): 

    run(tomo, alg)
    rmse_vec[i] = tomo.rmse()


if tomo.rank() == 0: 
    print('Reconstruction Complete, Saving Data..')
    print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = 'results/' + fName + '_' + alg 
meta = {'vol_size':vol_size,'Niter':Niter,'initAlg':initAlg}
results = {'rmse':rmse_vec}
mpi_save_results([fDir, fName], tomo, saveRecon, meta, results)

