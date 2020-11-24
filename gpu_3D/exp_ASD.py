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
(fName, tiltAngles, tiltSeries) = load_h5_data(vol_size,file_name)
(Nslice, Nray, Nproj) = tiltSeries.shape

# Initialize C++ Object.. 
tomo = mpi_astra_ctvlib.mpi_astra_ctvlib(Nslice, Nray, Nproj, np.deg2rad(tiltAngles))
initialize_algorithm(tomo, alg, initAlg)
tomo.initialize_recon_copy()

# Create Projections Vector
b = np.zeros([tomo.NsliceLoc(), Nray*Nproj])
for s in range(tomo.NsliceLoc()):
    b[s,:] = tiltSeries[s+tomo.firstSlice(),:,:].transpose().ravel()
tomo.set_tilt_series(b)

if tomo.rank() == 0: print('Starting Reconstruction')
dd_vec, tv_vec = np.zeros(Niter), np.zeros(Niter)
beta = beta0

#Main Loop
for i in tqdm(range(Niter)): 
    
    tomo.copy_recon() 

    run(tomo, alg, beta) # Reconstruction

    beta *= beta_red # Beta Reduction

    # # Measure Magnitude for TV - GD
    if (i == 0):
        dPOCS = tomo.matrix_2norm() * alpha
        dp = dPOCS / alpha
    else:
        dp = tomo.matrix_2norm()

    # Measure difference between exp / sim projections. 
    dd_vec[i] = tomo.data_distance()

    tomo.copy_recon()

    # # TV Minimization
    tv_vec[i] = tomo.tv_gd(ng, dPOCS)
    dg = tomo.matrix_2norm()

    if (dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

if tomo.rank() == 0: # Print Results
    print('Reconstruction Complete, Saving Data..')
    print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = 'results/' + fName + '_' + alg
meta = {'vol_size':vol_size,'Niter':Niter,'initAlg':initAlg,'beta':beta0,'beta_red':beta_red,'eps':eps}
meta.update({'alpha':alpha,'alpha_red':alpha_red})
results = {'tv':tv_vec,'dd':dd_vec}
mpi_save_results([fDir, fName], tomo, saveRecon, meta, results)

