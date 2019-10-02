# Script to simulate tomography reconstructions when a new projection 
# is added every XX Iterations.
# Used for simulated datasets (original volume / object is provided)

import sys, os
sys.path.append('./Utils')
from pytvlib import parallelRay, load_data
import plot_results as pr
import numpy as np
import ctvlib
import time
########################################

file_name = 'au_sto_tiltser.npy'

# Number of Iterations (TV Loop)
ng = 10

# ART Parameter.
beta0 = 0.1

# ART Reduction.
beta_red = 0.9

# Data Tolerance Parameter
eps = 0.02

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

#Amount of Iterations before next projection is collected.
Niter = 200

SNR = 100
noise = True
save = False
show_live_plot = 0

##########################################

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/au_sto_tiltAngles.npy')
Nproj = tiltAngles.shape[0]

#Read Image. 
(_, original_volume) = load_data(file_name)
file_name = 'au_sto'
(Nslice, Nray, _) = original_volume.shape

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

#Final vectors for dd, tv, and Niter. 
dd_vec = np.zeros([Nproj, Niter], dtype=np.float32)
tv_vec = np.zeros([Nproj, Niter], dtype=np.float32)
rmse_vec = np.zeros([Nproj, Niter], dtype=np.float32)

#Dynamic Tilt Series Loop. 
for i in range(Nproj):

    print('Reconstructing Tilt Angles: 1 -> ' + str(i+1) + ' / ' + str(Nproj))

    # Reset Beta.
    beta = beta0
 
    #Main Reconstruction Loop
    for j in range(Niter): 

        tomo_obj.copy_recon()

        #ART Reconstruction. 
        tomo_obj.ART(beta, i+1)

        #Positivity constraint 
        tomo_obj.positivity()

        #ART-Beta Reduction.
        beta *= beta_red

        #Forward Projection.
        tomo_obj.forwardProjection(i+1)

        #Measure Magnitude for TV - GD.
        if (i == 0 and j == 0):
            dPOCS0 = tomo_obj.matrix_2norm() * alpha
            dp = dPOCS0 / alpha
        else: # Measure change from ART.
            dp = tomo_obj.matrix_2norm() 

        if (j == 0):
    	    dPOCS = dPOCS0

        # Measure difference between exp/sim projections.
        dd_vec[i,j] = tomo_obj.dyn_vector_2norm(i+1)

        #Measure TV. 
        tv_vec[i,j] = tomo_obj.tv()

        #Measure RMSE.
        rmse_vec[i,j] = tomo_obj.rmse()

        tomo_obj.copy_recon() 

        #TV Minimization. 
        tomo_obj.tv_gd(ng, dPOCS)
        dg = tomo_obj.matrix_2norm()

        if (dg > dp * r_max and dd_vec[i,j] > eps):
            dPOCS *= alpha_red

    if save and (i+1)%15 == 0 :
        os.makedirs('Results/'+ file_name +'_Time/', exist_ok=True)
        recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
        for s in range(Nslice):
            recon[s,:,:] = tomo_obj.getRecon(s)
        np.save('Results/'+ file_name +'_Time/proj_' + str(i+1) + '_recon.npy', recon)

    if show_live_plot and (i+1) % 15 == 0:
        pr.sim_time_tv_live_plot(dd_vec,eps,tv_vec, tv0, rmse_vec, i)

#Save all the results to single matrix.
results = np.array([Niter, dd_vec, eps, tv_vec, tv0, rmse_vec])
os.makedirs('Results/'+ file_name +'_iter/', exist_ok=True)
np.save('Results/'+ file_name +'_iter/results.npy', results)

# Save the Reconstruction.
recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
for s in range(Nslice):
    recon[s,:,:] = tomo_obj.getRecon(s)
np.save('Results/'+ file_name +'_iter/final_recon.npy', recon)

# np.save('Results/'+ file_name +'_Time/gif.npy', gif)
