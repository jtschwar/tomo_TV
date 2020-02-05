# General 3D - ASD/TV Reconstruction with Positivity Constraint. 
# Intended for simulated datasets to measure RMSE and Volume's Original TV. 
# and to reconstruct large volume sizes (>1000^3) with Distributed Memory (OpenMPI)

import sys, os
sys.path.append('./Utils')
from pytvlib import parallelRay, timer, load_data
import numpy as np
import mpi_ctvlib 
import time
########################################

vol_size = '256_'
file_name = 'au_sto_tiltser.npy'

# Number of Iterations (Main Loop)
Niter = 10

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
save_recon = 1           # Save final Reconstruction. 
##########################################

#Read Image. (MPI_IO)
(file_name, original_volume) = load_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+ file_name +'_tiltAngles.npy', allow_pickle=True)
Nproj = tiltAngles.shape[0]

# Initialize C++ Object.. 
tomo_obj = mpi_ctvlib.mpi_ctvlib(Nslice, Nray, Nproj)
rank = tomo_obj.get_rank()
if rank == 0:
    print("Number of process: %d "%tomo_obj.get_nproc())
Nslice_loc = tomo_obj.get_Nslice_loc()
first_slice = tomo_obj.get_first_slice()

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
tomo_obj.load_A(A)
A = None
tomo_obj.rowInnerProduct()
if rank == 0:
    print('Measurement Matrix is Constructed!')

# If creating simulation with noise, set background value to 1.
if noise:
    original_volume[original_volume == 0] = 1

# Load Volume and Collect Projections. 
for s in range(Nslice_loc):
    tomo_obj.setOriginalVolume(original_volume[s+first_slice,:,:], s)
tomo_obj.create_projections()

# Apply poisson noise to volume.
if noise:
    tomo_obj.poissonNoise(SNR)

#Measure Volume's Original TV
tv0 = tomo_obj.original_tv()

dd_vec, tv_vec = np.zeros(Niter), np.zeros(Niter)
rmse_vec, time_vec = np.zeros(Niter), np.zeros(Niter)

counter = 1 

t0 = time.time()

#Main Loop
for i in range(Niter): 

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

    counter += 1
    time_vec[i] = time.time() - t0

    if ( i % 25 ==0 and rank == 0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))
        timer(t0, counter, Niter)

#Save all the results to single matrix.
if rank == 0:
    results = np.array([dd_vec, eps, tv_vec, tv0, rmse_vec, time_vec])
    os.makedirs('Results/'+ file_name +'_MPI/', exist_ok=True)
    np.save('Results/' + file_name + '_MPI/results.npy', results)

#Get and save the final reconstruction.
if save_recon: 
    recon_loc = np.zeros([Nslice_loc, Nray, Nray], dtype=np.float32, order='F')

    for s in range(Nslice_loc):
        recon_loc[s,:,:] = tomo_obj.getLocRecon(s+1)
    np.save('Results/{}_MPI/recon{}.npy'.format(file_name,rank), recon_loc)

    if rank == 0:
        import os
        recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F')

        s1 = 0
        for proc_num in range(tomo_obj.get_nproc()):

            temp_recon = np.load('Results/{}_MPI/recon{}.npy'.format(file_name,proc_num))
            s2 = temp_recon.shape[0] + s1
            recon[s1:s2,:,:] = temp_recon
            s1 = s2
            os.remove('Results/{}_MPI/recon{}.npy'.format(file_name,proc_num))

        np.save('Results/'+ file_name +'_MPI/recon.npy', recon)

tomo_obj.mpi_finalize()
