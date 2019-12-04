# General 3D - ASD/TV Reconstruction with Positivity Constraint. 
# Intended for simulated datasets to measure RMSE and Volume's Original TV. 
# and to reconstruct large volume sizes (>1000^3) with Distributed Memory (OpenMPI)

import sys, os
sys.path.append('../Utils')
from pytvlib import parallelRay, timer, load_data
import plot_results as pr
import numpy as np
import mpi_ctvlib 
import time
########################################

vol_size = '256_'
file_name = 'au_sto_tiltser.npy'

# Number of Iterations (Main Loop)
Niter = 300

# Number of Iterations (TV Loop)
ng = 10

# Parameter in ART Reconstruction.
beta = 0.25

# ART Reduction.
beta_red = 0.985

# Data Tolerance Parameter
eps = 0.019

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

SNR = 100

#Outcomes:
noise = True                # Add noise to the reconstruction.
show_live_plot = 0
save_recon = 0           # Save final Reconstruction. 
##########################################

#Read Image. 
(file_name, original_volume) = load_data(vol_size,file_name)
# this has some issue
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

print('Measurement Matrix is Constructed!')

# If creating simulation with noise, set background value to 1.
if noise:
    original_volume[original_volume == 0] = 1

# Load Volume and Collect Projections. 
for s in range(tomo_obj.Nslice_loc):
    tomo_obj.setOriginalVolume(original_volume[s+tomo_obj.first_slice,:,:], s)
tomo_obj.create_projections()

# Apply poisson noise to volume.
if noise:
    tomo_obj.poissonNoise(SNR)

#Measure Volume's Original TV
tv0 = tomo_obj.original_tv()

gif = np.zeros([Nray, Nray, Niter], dtype=np.float32)
gif2 = np.zeros([Nray, Nray, Niter], dtype=np.float32)

dd_vec = np.zeros(Niter)
tv_vec = np.zeros(Niter)
rmse_vec = np.zeros(Niter)
time_vec = np.zeros(Niter)

counter = 1 

t0 = time.time()

#Main Loop
for i in range(Niter): 

    if ( i % 25 ==0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    tomo_obj.copy_recon()

    #ART Reconstruction. 
    tomo_obj.sART(beta, -1)

    #Positivity constraint 
    tomo_obj.positivity()

    #ART-Beta Reduction
    beta *= beta_red 

    #Forward Projection.
    tomo_obj.forwardProjection(-1)

    #Measure Magnitude for TV - GD.
    if (i == 0):
        dPOCS = tomo_obj.matrix_2norm() * alpha
        dp = dPOCS / alpha
    else: # Measure change from ART.
        dp = tomo_obj.matrix_2norm() 

    # Measure difference between exp/sim projections.
    dd_vec[i] = tomo_obj.vector_2norm()

    #Measure TV. 
    tv_vec[i] = tomo_obj.tv()

    #Measure RMSE.
    rmse_vec[i] = tomo_obj.rmse()

    tomo_obj.copy_recon() 

    #TV Minimization. 
    tomo_obj.tv_gd(ng, dPOCS)
    dg = tomo_obj.matrix_2norm()

    if(dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

    if (i+1)% 25 == 0:
        timer(t0, counter, Niter)
        if show_live_plot:
            pr.sim_ASD_live_plot(dd_vec, eps, tv_vec, tv0, rmse_vec, i)
    counter += 1
    time_vec[i] = time.time() - t0

#Save all the results to single matrix.
results = np.array([dd_vec, eps, tv_vec, tv0, rmse_vec, time_vec])
os.makedirs('Results/'+ file_name +'_ASD/', exist_ok=True)
np.save('Results/' + file_name + '_ASD/results5.npy', results)

#Get and save the final reconstruction.
if save_recon: 
    recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 

    for s in range(Nslice):
        recon[s,:,:] = tomo_obj.getRecon(s)
    
    np.save('Results/TV_'+ file_name + '_recon.npy', recon)
