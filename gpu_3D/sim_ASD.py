# General 3D - ASD/TV Reconstruction with Positivity Constraint. 
# Intended for simulated datasets to measure RMSE and Volume's Original TV. 

import mpi_astra_ctvlib
from tqdm import tqdm
from pytvlib import *
import numpy as np
check_cuda()
########################################

vol_size = '256'
file_name = 'au_sto.h5'

# Algorithm
alg = 'SART'
initAlg = 'random' # Algorithm Parameters (ie Projection Order or Filter)

# Number of Iterations (Main Loop)
Niter = 100

# Descent Parameter and Reduction
beta0, beta_red = 0.25, 0.9985

# Data Tolerance Parameter
eps = 0.025

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2
ng = 10

# Save Final Reconstruction. 
saveRecon = True

##########################################

#Read Image. 
(fName, tiltAngles, original_volume) = load_h5_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape
Nproj = tiltAngles.shape[0]

# Initialize C++ Object.. 
tomo_obj = mpi_astra_ctvlib.mpi_astra_ctvlib(Nslice, Nray, Nproj, np.deg2rad(tiltAngles))
tomo_obj.initializeInitialVolume()
tomo_obj.initializeReconCopy()

# Load Volume and Collect Projections. 
for s in range(tomo_obj.NsliceLoc()):
    tomo_obj.setOriginalVolume(original_volume[s+tomo_obj.firstSlice(),:,:], s)

initialize_algorithm(tomo_obj, alg, initAlg)

# Add Poisson Noise to Volume
tomo_obj.set_background()
tomo_obj.create_projections()
tomo_obj.poissonNoise(75)

#Measure Volume's Original TV
tv0 = tomo_obj.original_tv()

if tomo_obj.rank() == 0: print('Starting Reconstruction')
rmse_vec, dd_vec, tv_vec = np.zeros(Niter), np.zeros(Niter), np.zeros(Niter)
beta = beta0

#Main Loop
for i in tqdm(range(Niter)): 
    
    tomo_obj.copy_recon() 

    run(tomo_obj, alg, beta) # Reconstruction

    beta *= beta_red # Beta Reduction

    tomo_obj.forwardProjection()

    # # Measure Magnitude for TV - GD
    if (i == 0):
        dPOCS = tomo_obj.matrix_2norm() * alpha
        dp = dPOCS / alpha
    else:
        dp = tomo_obj.matrix_2norm()

    # Measure difference between exp / sim projections. 
    dd_vec[i] = tomo_obj.vector_2norm()

    rmse_vec[i] = tomo_obj.rmse() # Measure RMSE

    tomo_obj.copy_recon()

    # # TV Minimization
    tv_vec[i] = tomo_obj.tv_gd(ng, dPOCS)
    dg = tomo_obj.matrix_2norm()

    if (dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

if tomo_obj.rank() == 0: # Print Results
    print('RMSE: ' , rmse_vec)
    print('(TV0: ', tv0,')TV: '   , tv_vec)
    print('DD: '   , dd_vec)
    print('Reconstruction Complete, Saving Data..')
    print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = 'results/' + fName + '_' + alg
meta = {'vol_size':vol_size,'Niter':Niter,'initAlg':initAlg,'beta':beta0,'beta_red':beta_red,'eps':eps}
meta.update({'alpha':alpha,'alpha_red':alpha_red,'tv0':tv0})
results = {'rmse':rmse_vec,'tv':tv_vec,'dd':dd_vec}
mpi_save_results([fDir, fName], tomo_obj, saveRecon, meta, results)

