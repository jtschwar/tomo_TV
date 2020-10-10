# General 3D - SIRT Reconstruction with Positivity Constraint. 

from pytvlib import *
import astra_ctvlib
import numpy as np
import time
from tqdm import tqdm
check_cuda()
########################################

vol_size = '256'
file_name = 'au_sto.h5'

# Number of Iterations (Main Loop)
Niter = 100

# Algorithm
alg = 'SIRT'
initAlg = '' # Algorithm Parameters (ie Projection Order or FBF-Filter)

# Descent Parameter and Reduction
beta = 0.5
beta_red = 0.995

# Save Final Reconstruction. 
saveRecon = True

##########################################

#Read Image. 
(fName, tiltAngles, original_volume) = load_h5_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape
Nproj = tiltAngles.shape[0]

print('Loaded h5 file, now intiializing c++ object')

# Initialize C++ Object.. 
tomo_obj = astra_ctvlib.astra_ctvlib(Nslice, Nray, Nproj, np.deg2rad(tiltAngles))
tomo_obj.initilizeInitialVolume()

# Load Volume and Collect Projections. 
for s in range(Nslice):
    tomo_obj.setOriginalVolume(original_volume[s,:,:],s)

print('Loaded Test Object, now creating projections')

initialize_algorithm(tomo_obj, alg, initAlg)
tomo_obj.create_projections()

print('Starting Reconstruction')
rmse_vec = np.zeros(Niter)

#Main Loop
for i in tqdm(range(Niter)): 

    run(tomo_obj, alg)

    rmse_vec[i] = tomo_obj.rmse()

print('RMSE: ', rmse_vec)
print('Reconstruction Complete, Saving Data..')
print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_' + alg
meta = {'vol_size':vol_size,'Niter':Niter, 'initAlg':initAlg} # Metadata / Convergence Parameters
results = {'rmse':rmse_vec} # Convergence Results (i.e. DD / TV / RMSE)
save_results([fDir,fName], meta, results)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)
