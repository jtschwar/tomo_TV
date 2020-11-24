# General 3D - SIRT Reconstruction with Positivity Constraint. 

from pytvlib import *
from tqdm import tqdm
import numpy as np
import ctvlib 
import time
########################################

vol_size = '256_'
file_name = 'au_sto_tiltser.npy'

# Algorithm and Projection Order
alg = 'ART'

# Number of Iterations (Main Loop)
Niter = 100

# Descent Parameter (for ART / SART)
beta0, beta_red = 0.5, 0.985

# Data Tolerance Parameter
eps = 0.02

# TV Minimizatino Parameter
alpha, alpha_red = 0.2, 0.95

# Reduction Criteria
r_max = 0.95
ng = 10
SNR = 100

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

# Initialize Additional Volumes for ASD_POCS Algorithm
tomo.initialize_tv_recon(); tomo.initialize_recon_copy()

# Create Projections
create_projections(tomo, original_volume, SNR)

# Measure Convergence Properties
tv0 = tomo.original_tv()
rmse_vec, dd_vec, tv_vec = np.zeros(Niter), np.zeros(Niter), np.zeros(Niter)

#Main Loop
print('Starting Reconstruction')
for i in tqdm(range(Niter)): 

    tomo.copy_recon()

    run(tomo, alg, beta)

    #ART-Beta Reduction.
    if alg != 'SIRT': beta *= beta_red

    if (i == 0):
        dPOCS = tomo.matrix_2norm() * alpha
        dp = dPOCS / alpha
    else:
        dp = tomo.matrix_2norm()

    # Measure Data Distance (DD)
    dd_vec[i] = tomo.data_distance()

    # Measure RMSE
    rmse_vec[i] = tomo.rmse()

    tomo.copy_recon()

    # # TV Minimization
    tv_vec[i] = tomo.tv()
    tomo.tv_gd(ng, dPOCS)
    dg = tomo.matrix_2norm()

    if (dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

print('Reconstruction Complete, Saving Data..')
print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_ASD' 
meta = {'Niter':Niter,'ng':ng,'beta':beta0,'beta_red':beta_red,'eps':eps,'r_max':r_max}
meta.update({'alpha':alpha,'alpha_red':alpha_red,'SNR':SNR,'vol_size':vol_size, 'tv0':tv0})
results = {'rmse':rmse_vec,'tv':tv_vec,'dd':dd_vec}
save_results([fDir,fName], meta, results, tomo, saveRecon)
