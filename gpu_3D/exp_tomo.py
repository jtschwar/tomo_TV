# General 3D - SIRT Reconstruction with Positivity Constraint. 

from pytvlib import *
import astra_ctvlib
import numpy as np
import time
from tqdm import tqdm
check_cuda()
########################################

# File Name
vol_size = '256'
file_name = 'au_sto.h5'

# Number of Iterations (Main Loop)
Niter = 100

alg = 'SIRT' # Algorithm
initAlg = '' # Algorithm Parameters (ie Projection Order or Filter)

# Descent Parameter and Reduction
beta0 = 0.5
beta_red = 0.995

# Save Final Reconstruction. 
saveRecon = True

##########################################

#Read Image. 
#(fName, tiltSeries) = load_data(vol_size,file_name)
#tiltAngles = np.load('Tilt_Series/'+fName+'_tiltAngles.npy')
(fName, tiltAngles, tiltSeries) = load_h5_data(vol_size,file_name)
(Nslice, Nray, Nproj) = tiltSeries.shape
Nproj = tiltAngles.shape[0]

b = np.zeros([Nslice, Nray*Nproj])
for s in range(Nslice):
    b[s,:] = tiltSeries[s,:,:].transpose().ravel()

print('Loaded h5 file, now intiializing c++ object')

# Initialize C++ Object.. 
tomo_obj = astra_ctvlib.astra_ctvlib(Nslice, Nray, Nproj, np.deg2rad(tiltAngles))
initialize_algorithm(tomo_obj, alg, initAlg)
tomo_obj.setTiltSeries(b)

dd_vec = np.zeros(Niter)
beta = beta0

print('Starting Reconstruction')

#Main Loop
for i in tqdm(range(Niter)): 

    run(tomo_obj, alg, beta)

    # Data Distance
    tomo_obj.forwardProjection()
    dd_vec[i] = tomo_obj.vector_2norm()

print('Reconstruction Complete, Saving Data..')
print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_' + alg
meta = {'vol_size':vol_size,'Niter':Niter, 'initAlg':initAlg, 'beta':beta, 'beta_red': beta_red}
results = {'rmse':dd_vec}
save_results([fDir,fName], meta, results)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)
