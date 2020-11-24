# General 3D - SIRT Reconstruction with Positivity Constraint. 

from pytvlib import *
import numpy as np
import ctvlib 
import time
########################################

vol_size = '256_'
file_name = 'au_sto_tiltser.npy'

# Number of Iterations (Main Loop)
Niter = 100

alg = 'SIRT'

# Descent Parameter (for ART / SART)
beta0, beta_red = 0.5, 0.985

# Save Final Reconstruction. 
saveRecon = True

##########################################

# Read Image. 
(fName, tiltSeries) = load_data(vol_size,file_name)
(Nslice, Nray, Nproj) = tiltSeries.shape

# Initialize C++ Object.. 
tomo = ctvlib.ctvlib(Nslice, Nray, Nproj)

# Pass the Tilt Series to C++
load_exp_tilt_series(tomo, tiltSeries)
tiltSeries = None

# Read Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+fName+'_tiltAngles.npy')
Nproj = tiltAngles.shape[0]

# Generate measurement matrix
initialize_algorithm(tomo, alg, Nray, tiltAngles)
if alg == 'SIRT': beta = 1/tomo.lipschits()
print('Measurement Matrix is Constructed!')

dd_vec = np.zeros(Niter)

#Main Loop
for i in tqdm(range(Niter)): 

    run(tomo, alg, beta)

    #ART-Beta Reduction.
    if alg != 'SIRT': beta *= beta_red

    # Measure Data Distance (DD)
    dd_vec[i] = tomo.data_distance()


print('Reconstruction Complete, Saving Data..')
print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_' + alg
meta = {'vol_size':vol_size,'Niter':Niter,'beta':beta, 'beta_red':beta_red}
results = {'dd':dd_vec}
save_results([fDir,fName], meta, results, tomo, saveRecon)
