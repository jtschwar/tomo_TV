# General 3D - ART Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
from pytvlib import *
import astra_ctvlib
import numpy as np
import time
from tqdm import tqdm
check_cuda()
########################################

vol_size = '256_'
file_name = 'au_sto.h5'

# Number of Iterations (Main Loop)
Niter = 5 

# Parameter in ART Reconstruction.
beta0 = 1.0

# ART Reduction.
beta_red = 0.995

#Save Final Reconstruction
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

tomo_obj.create_projections()

#tomo_obj.initializeSART('sequential')
tomo_obj.initializeSART('random')

beta = beta0
rmse_vec = np.zeros(Niter)

print('Starting Reconstruction')

#Main Loop
for i in tqdm(range(Niter)): 

    tomo_obj.SART(beta0)

    #ART-Beta Reduction
    beta *= beta_red 

    rmse_vec[i] = tomo_obj.rmse()

print(rmse_vec)

print('Reconstruction Complete, Saving Data..')
print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_ART'
meta = {'vol_size':vol_size,'Niter':Niter,'beta':beta0,'beta_red':beta_red}
results = {'rmse':rmse_vec}
save_results([fDir,fName], meta, results)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)
