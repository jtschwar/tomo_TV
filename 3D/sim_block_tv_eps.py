# Script to simulate tomography reconstructions when a new projection 
# is added every XX Iterations. Instead of Iter_TV that runs each projection 
# for a fixed number of iterations (sequentially), here we run a single projection
# until convergence (~125 iterations) and use the time elapsed to determine
# how many more projections to add/ ('sample').
# Used for simulated datasets (original volume / object is provided)

import sys, os
sys.path.append('./Utils')
from pytvlib import parallelRay, load_data
import numpy as np
import ctvlib
import time
########################################

vol_size = '512_'
file_name = 'au_sto_tiltser.npy'

# Number of Iterations (TV Loop)
ng = 10

# ART Parameter.
beta0 = 0.25

# ART Reduction.
beta_red = 0.98

# Data Tolerance Parameter
eps_final = 0.019

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

#Amount of time before next projection is collected (Seconds).
time_limit = 180

# Max Number of iterations before next projection is collected. 
max_iter = 125

#SNR
SNR = 100
noise = True

save = False        # Save final reconstruction. 
show_live_plot = 1  # Show intermediate results. 

##########################################

#Read Image. 
(file_name, original_volume) = load_data(vol_size,file_name)
file_name = 'au_sto'
(Nslice, Nray, _) = original_volume.shape

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+ file_name +'_tiltAngles.npy')
Nproj = tiltAngles.shape[0]

# Initialize C++ Object.. 
tomo_obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
tomo_obj.load_A(A)
A = None
tomo_obj.rowInnerProduct()

# If creating simulation with noise, set background value to 1.
if noise:
    original_volume[original_volume == 0] = 1

# Load Volume and Collect Projections. 
for s in range(Nslice):
    tomo_obj.setOriginalVolume(original_volume[s,:,:],s)
tomo_obj.create_projections()

# Apply poisson noise to volume.
if noise:
    tomo_obj.poissonNoise(SNR)

tv0 = tomo_obj.original_tv()

# gif = np.zeros([Nray, Nray, Nproj], dtype=np.float32)

#Final vectors for dd, tv, and Niter. 
Niter = np.zeros(Nproj, dtype=np.int32)
fdd_vec, ftv_vec, frmse_vec = np.array([]), np.array([]), np.array([])
ftime_vec, fnabla_dd_vec, eps_vec = np.array([]), np.array([]),np.array([])

Niter_est = 100

i = 0 # Projection Number
max_time = time_limit * (Nproj-1)
t_start = time.time()

#Dynamic Tilt Series Loop. 
while i < Nproj:

    print('Reconstructing Tilt Angles: 0 -> ' + str(i+1) + ' / ' + str(Nproj))

    # Reset Beta.
    beta = beta0 * (1.0 - 5/6 * i/Nproj)

    dd_vec = np.zeros(Niter_est, dtype=np.float32)
    nabla_dd_vec = np.zeros(Niter_est, dtype=np.float32)
    tv_vec = np.zeros(Niter_est, dtype=np.float32)
    rmse_vec = np.zeros(Niter_est, dtype=np.float32)
    time_vec = np.zeros(Niter_est, dtype=np.float32)

    t0 = time.time()

    if i != Nproj - 1:
        eps = eps_final + 0.001 * (Nproj / (i + 1))
 
    #Main Reconstruction Loop
    while True: 

        tomo_obj.copy_recon()

        #ART Reconstruction. 
        tomo_obj.sART(beta, i+1)

        #Positivity constraint 
        tomo_obj.positivity()

        #ART-Beta Reduction.
        beta *= beta_red

        #Forward Projection.
        tomo_obj.forwardProjection(i+1)

        #Measure Magnitude for TV - GD.
        if (Niter[0] == 0):
            dPOCS0 = tomo_obj.matrix_2norm() * alpha
            dp = dPOCS0 / alpha
        else: # Measure change from ART.
            dp = tomo_obj.matrix_2norm() 

        if (Niter[i] == 0):
    	    dPOCS = dPOCS0

        # Measure difference between exp/sim projections.
        dd_vec[Niter[i]] = tomo_obj.dyn_vector_2norm(i+1)
        if i > 0: #Measure change in DD. 
            nabla_dd_vec[Niter[i]] = dd_vec[Niter[i]] - dd_vec[Niter[i-1]] 

        #Measure TV. 
        tv_vec[Niter[i]] = tomo_obj.tv()

        #Measure RMSE.
        rmse_vec[Niter[i]] = tomo_obj.rmse()

        tomo_obj.copy_recon() 

        #TV Minimization. 
        tomo_obj.tv_gd(ng, dPOCS)
        dg = tomo_obj.matrix_2norm()

        if (dg > dp * r_max and dd_vec[Niter[i]] > eps):
            dPOCS *= alpha_red

        time_vec[Niter[i]] = time.time() - t_start

        Niter[i] += 1

        #Calculate current time. 
        ctime= time.time() - t0
        t_time = time.time() - t_start

        # Check convergence criteria.
        if (ctime > time_limit and Niter[i] > max_iter-1) or t_time > max_time:
            break

    # Get a slice for visualization of convergence. 
    # gif[:,:,i] = tomo_obj.getRecon(259)

    Niter_est = Niter[i]
    print('Number of Iterations: ' + str(Niter[i]) + '\n')

    #Remove Excess elements.
    dd_vec = dd_vec[:Niter[i]]
    tv_vec = tv_vec[:Niter[i]]
    rmse_vec = rmse_vec[:Niter[i]]
    time_vec = time_vec[:Niter[i]]
    nabla_dd_vec = nabla_dd_vec[:Niter[i]]

    #Append to final vector. 
    fdd_vec = np.append(fdd_vec, dd_vec)
    ftv_vec = np.append(ftv_vec, tv_vec)
    frmse_vec = np.append(frmse_vec, rmse_vec)
    ftime_vec = np.append(ftime_vec, time_vec)
    eps_vec = np.append(eps_vec, eps)
    fnabla_dd_vec = np.append(fnabla_dd_vec, nabla_dd_vec)

    # Determine how many more projections to add w time elapsed.
    ctime= time.time() - t_start 
    i = int(ctime/time_limit)

    if i == Nproj - 1:
        max_time *= 10

#Save all the results to single matrix.
results = np.array([Niter, ftime_vec, fdd_vec, eps_vec, ftv_vec, tv0, frmse_vec])
os.makedirs('Results/'+ file_name +'_block_eps/', exist_ok=True)
np.save('Results/'+ file_name +'_block_eps/results.npy', results)
np.save('Results/'+ file_name +'_block_eps/nabla_dd.npy', fnabla_dd_vec)
# np.save('Results/'+ file_name +'_block_eps/gif.npy', gif)

if save:
    # Save the Reconstruction.
    recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
    for s in range(Nslice):
        recon[s,:,:] = tomo_obj.getRecon(s)
    np.save('Results/'+ file_name +'_block_eps/final_recon.npy', recon)
