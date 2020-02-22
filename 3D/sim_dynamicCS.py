# Script to simulate tomography reconstructions when a new projection 
# is added every XX Iterations. Instead of Iter_TV that runs each projection 
# for a fixed number of iterations (sequentially), here we run a single projection
# until convergence (~125 iterations) and use the time elapsed to determine
# how many more projections to add/ ('sample').
# Used for simulated datasets (original volume / object is provided)

import sys, os
sys.path.append('./Utils')
import plot_results as pr
from pytvlib import *
import numpy as np
import ctvlib
import time
########################################

vol_size = '256_'
file_name = 'Co2P_tiltser.tif'

# Number of Iterations (TV Loop)
ng = 10

# ART Parameter.
beta0 = 0.25

# ART Reduction.
beta_red = 0.98

# Data Tolerance Parameter
eps = 0.019

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

#Amount of time before next projection is collected (Seconds).
time_limit = 180

# Max Number of iterations before next projection is collected. 
max_iter = 125

SNR = 100

#Outcomes:
show_live_plot = 0
saveGif, saveRecon = True, True           
gif_slice = 156
##########################################

#Read Image. 
(file_name, original_volume) = load_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape


# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+file_name+'_tiltAngles.npy')
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
if SNR != 0:
    original_volume[original_volume == 0] = 1

# Load Volume and Collect Projections. 
for s in range(Nslice):
    tomo_obj.setOriginalVolume(original_volume[s,:,:],s)
tomo_obj.create_projections()

# Apply poisson noise to volume.
if SNR != 0:
    tomo_obj.poissonNoise(SNR)

#Final vectors for dd, tv, and Niter. 
tv0 = tomo_obj.original_tv()
gif = np.zeros([Nray, Nray, Nproj], dtype=np.float32)
Niter = np.zeros(Nproj, dtype=np.int32)
fdd_vec,ftv_vec  = np.array([]),np.array([])
frmse_vec,ftime_vec = np.array([]),np.array([])

i = 0 # Projection Number
max_time = time_limit * (Nproj-1)
t_start = time.time()

#Dynamic Tilt Series Loop. 
while i < Nproj:

    print('Reconstructing Tilt Angles: 0 -> ' + str(i+1) + ' / ' + str(Nproj))

    # Reset Beta.
    beta = beta0 * (1.0 - 2/3 * i/Nproj)

    dd_vec,tv_vec = np.zeros(max_iter, dtype=np.float32), np.zeros(max_iter, dtype=np.float32)
    rmse_vec,time_vec = np.zeros(max_iter, dtype=np.float32), np.zeros(max_iter, dtype=np.float32)

    t0 = time.time()
 
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
        ctime =  time.time() - t0  
        t_time = time.time() - t_start

        # Check convergence criteria.
        if (ctime > time_limit and Niter[i] > max_iter-1) or t_time > max_time:
            break

    # Get a slice for visualization of convergence. 
    gif[:,:,i] = tomo_obj.getRecon(gif_slice)

    Niter_est = Niter[i]
    print('Number of Iterations: ' + str(Niter[i]) + '\n')

    #Remove Excess elements.
    dd_vec = dd_vec[:Niter[i]]
    tv_vec = tv_vec[:Niter[i]]
    rmse_vec = rmse_vec[:Niter[i]]
    time_vec = time_vec[:Niter[i]]

    #Append to final vector. 
    fdd_vec = np.append(fdd_vec, dd_vec)
    ftv_vec = np.append(ftv_vec, tv_vec)
    frmse_vec = np.append(frmse_vec, rmse_vec)
    ftime_vec = np.append(ftime_vec, time_vec)

    # Determine how many more projections to add w time elapsed.
    ctime= time.time() - t_start 
    i = int(ctime/time_limit)

    if i == Nproj - 1:
        max_time *= 10

    # Show live plot. 
    if show_live_plot and (i+1) % 15 == 0:
        pr.sim_time_tv_live_plot(fdd_vec,eps,ftv_vec, tv0, frmse_vec, Niter,i)

print('Reconstruction Complete, Saving Data..')
print('Save Gif :: {}, Save Recon :: {}'.format(saveGif, saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_dynamic_CS'
meta = {'ng':ng,'beta':beta0,'beta_red':beta_red,'eps':eps,'r_max':r_max,'max_iter':max_iter}
meta.update({'alpha':alpha,'alpha_red':alpha_red,'SNR':SNR,'vol_size':vol_size,'time_limit':time_limit})
results = {'dd':fdd_vec,'eps':eps,'tv':ftv_vec,'tv0':tv0,'rmse':frmse_vec,'time':ftime_vec}
results.update({'Niter':Niter})
save_results([fDir,fName], meta, results)

if saveGif: 
    save_gif([fDir,fName], gif_slice, gif)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)
