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
import argparse
import ctvlib
import time

parser = argparse.ArgumentParser()
parser.add_argument("--beta0", "-b",type=float,default=0.15)
parser.add_argument("--Niter",type=int,default=500)
args = parser.parse_args()

########################################

vol_size = '256_'
file_name = 'au_sto_tiltser.npy'

Niter = args.Niter

# Number of Iterations (TV Loop)
ng = 10

# ART Parameter.
beta0 = args.beta0

# ART Reduction.
beta_red = 0.98

# Data Tolerance Parameter
eps_max = 0.04
eps_min = 0.015
deps = 0.0025

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

noise = True
SNR = 75

save = False        # Save final reconstruction. 

##########################################

#Read Image. 
(file_name, original_volume) = load_data(vol_size,file_name)
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

#Measure Volume's Original TV
tv0 = tomo_obj.original_tv()

gif = np.zeros([Nray, Nray, Nproj], dtype=np.float32)

neps = np.ceil((eps_max-eps_min)/deps) + 1
eps = np.linspace(eps_min, eps_max, neps)
eps = eps[::-1]
eps = np.append(eps,eps[-2:0:-1])

#Final vectors for dd, tv, and Niter. 
dd_vec, tv_vec, rmse_vec = np.zeros([len(eps), Niter]), np.zeros([len(eps), Niter]), np.zeros([len(eps), Niter])

ii = 0
#Dynamic Tilt Series Loop. 
for ee in eps:

    ee = round(ee,4)

    print('Simulating Epsilon: {}, for Iteration: {}/{}'.format(ee,ii+1, len(eps)))

    # Reset Beta. 
    beta = beta0

    if ee < 0.0225:
        beta = beta0 * 4
        print('Setting beta to : {}'.format(beta))

    #Main Reconstruction Loop
    for jj in range(Niter): 

        tomo_obj.copy_recon()

        #ART Reconstruction. 
        tomo_obj.sART(beta, -1)

        #Positivity constraint 
        tomo_obj.positivity()

        #ART-Beta Reduction.
        beta *= beta_red

        #Forward Projection.
        tomo_obj.forwardProjection(-1)

        #Measure Magnitude for TV - GD.
        if (ii == 0 or (jj == 1 and ii != 0)):
            dPOCS = tomo_obj.matrix_2norm() * alpha
            dp = dPOCS / alpha
        else: # Measure change from ART.
            dp = tomo_obj.matrix_2norm() 

        # Measure difference between exp/sim projections.
        dd_vec[ii,jj] = tomo_obj.vector_2norm() 

        #Measure TV. 
        tv_vec[ii,jj] = tomo_obj.tv()

        #Measure RMSE.
        rmse_vec[ii,jj] = tomo_obj.rmse()

        tomo_obj.copy_recon() 

        #TV Minimization. 
        tomo_obj.tv_gd(ng, dPOCS)
        dg = tomo_obj.matrix_2norm()

        if (dg > dp * r_max and dd_vec[ii,jj] > ee):
            dPOCS *= alpha_red

    # Niter = Niter0
    ii += 1

    # Get a slice for visualization of convergence. 
    gif[:,:,ii] = tomo_obj.getRecon(129)

    #Save all the results to single matrix.
    results = np.array([dd_vec, eps, tv_vec, tv0, rmse_vec])
    os.makedirs('Results/'+ file_name +'_eps_sim/', exist_ok=True)
    np.save('Results/'+ file_name +'_eps_sim/results'+str(beta0)+'_'+str(Niter)+'.npy', results)
    np.save('Results/'+ file_name +'_eps_sim/gif'+str(beta0)+'_'+str(Niter)+'.npy', gif)


# #Save all the results to single matrix.
# results = np.array([dd_vec, eps, tv_vec, tv0, rmse_vec])
# os.makedirs('Results/'+ file_name +'_eps_sim/', exist_ok=True)
# np.save('Results/'+ file_name +'_eps_sim/results'+str(beta0)+'_'+str(Niter)+'.npy', results)
# np.save('Results/'+ file_name +'_eps_sim/gif'+str(beta0)+'_'+str(Niter)+'.npy', gif)

if save:
    # Save the Reconstruction.
    recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
    for s in range(Nslice):
        recon[s,:,:] = tomo_obj.getRecon(s)
    np.save('Results/'+ file_name +'_eps_sim/final_recon.npy', recon)
