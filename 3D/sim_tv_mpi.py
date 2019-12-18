# General 3D - ASD/TV Reconstruction with Positivity Constraint. 
# Intended for simulated datasets to measure RMSE and Volume's Original TV. 
# and to reconstruct large volume sizes (>1000^3) with Distributed Memory (OpenMPI)

import sys, os
sys.path.append('./Utils')
from pytvlib import parallelRay, timer, load_data
#from mpi4py import MPI
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

# Initalize pyMPI 
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()

#Read Image. (MPI_IO)
(file_name, original_volume) = load_data(vol_size,file_name)
file_name = 'au_sto'
(Nslice, Nray, _) = original_volume.shape

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+ file_name +'_tiltAngles.npy', allow_pickle=True)
Nproj = tiltAngles.shape[0]

# Initialize C++ Object.. 
tomo_obj = mpi_ctvlib.mpi_ctvlib(Nslice, Nray, Nproj)
if tomo_obj.get_rank()==0:
    print("Number of process: %d "%tomo_obj.get_nproc())
Nslice_loc = tomo_obj.get_Nslice_loc()
first_slice = tomo_obj.get_first_slice()
# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
tomo_obj.load_A(A)
A = None
tomo_obj.rowInnerProduct()
if tomo_obj.get_rank()==0:
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
    if ( i % 1 ==0 and tomo_obj.get_rank()==0):
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

    counter += 1
    time_vec[i] = time.time() - t0

    if (tomo_obj.get_rank()==0):
        print("rmse: %s" %rmse_vec[i])

    #Save all the results to single matrix.

if tomo_obj.get_rank() == 0:
    results = np.array([dd_vec, eps, tv_vec, tv0, rmse_vec, time_vec])
    os.makedirs('Results/'+ file_name +'_MPI/', exist_ok=True)
    np.save('Results/' + file_name + '_MPI/results.npy', results)
    print("tv_vec", tv_vec)
    print("time_vec", time_vec)
    print("dd_vec", dd_vec)

#Get and save the final reconstruction.
tomo_obj.gather_recon()
if save_recon and tomo_obj.get_rank()==0: 
    recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F')
    #Use mpi4py to do MPI_Gatherv
    for s in range(Nslice):
        # recon_loc[s+first_slice,:,:] = tomo_obj.getRecon(s)
        recon[s,:,:] = tomo_obj.getRecon(s)
    np.save('Results/TV_'+ file_name + '_recon.npy', recon)
tomo_obj.mpi_finalize()
