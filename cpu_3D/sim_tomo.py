# General 3D - SIRT Reconstruction with Positivity Constraint. 

from pytvlib import *
from tqdm import tqdm
import numpy as np
import ctvlib 
import time
########################################

vol_size = '256_'
file_name = 'au_sto_tiltser.npy'

# Number of Iterations (Main Loop)
Niter = 100

alg = 'randART'

# Descent Parameter (for ART / randART)
beta0, beta_red = 0.5, 0.995

# Save Final Reconstruction. 
saveRecon = True

##########################################

# Read Image. 
(fName, original_volume) = load_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape

# Read Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+fName+'_tiltAngles.npy')
Nproj = tiltAngles.shape[0]

# Initialize C++ Object.. 
tomo = ctvlib.ctvlib(Nslice, Nray, Nproj)

# Generate measurement matrix
initialize_algorithm(tomo, alg, Nray, tiltAngles)
print('Measurement Matrix is Constructed!')
if alg == 'SIRT': beta0 = 1/tomo.lipschits()
beta = beta0

# Create Projections
create_projections(tomo, original_volume)

dd_vec, rmse_vec = np.zeros(Niter), np.zeros(Niter)

#Main Loop
print('Starting Reconstruction')
for i in tqdm(range(Niter)): 

    run(tomo, alg, beta)

    #ART-Beta Reduction.
    if alg != 'SIRT': beta *= beta_red

    # Measure Data Distance (DD)
    dd_vec[i] = tomo.data_distance()

    # Measure RMSE
    rmse_vec[i] = tomo.rmse()


print('Reconstruction Complete, Saving Data..')
print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_' + alg
meta = {'vol_size':vol_size,'Niter':Niter,'beta':beta0, 'beta_red':beta_red}
results = {'dd':dd_vec, 'rmse':rmse_vec}
save_results([fDir,fName], meta, results, tomo, saveRecon)
