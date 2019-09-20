# Script to simulate tomography reconstructions when a new projection 
# is added every three minutes.
# Used for Experimental Datasets (projections are provided. )

import sys, os
sys.path.append('./Utils')
from pytvlib import parallelRay, load_data
import plot_results as pr
import numpy as np
import ctvlib
import time
########################################

file_name = '512_Co2P_tiltser.tif'

# Number of Iterations (TV Loop)
ng = 10

# ART Parameter.
beta0 = 1.0

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 0.25

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

#Amount of time before next projection is collected (Seconds).
time_limit = 15

save = True
show_final_plot = False
show_live_plot = True

##########################################

#Read Image. 
(file_name, tiltSeries) = load_data(file_name)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros( [Nslice, Nray*Nproj] )

# Initialize C++ Object.. 
tomo_obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

#Transfer Tilt Series to C++ Object. 
for s in range(Nslice):
    b[s,:] = tiltSeries[s,:,:].transpose().ravel()
tomo_obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+ file_name +'_tiltAngles.npy')

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
tomo_obj.load_A(A)
A = None
tomo_obj.rowInnerProduct()

#Final vectors for dd, tv, and Niter. 
recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
Niter = np.zeros(Nproj, dtype=np.int32)
fdd_vec = np.array([])
ftv_vec = np.array([])
Niter_est = 50

#Dynamic Tilt Series Loop. 
for i in range(Nproj):

    print('Reconstructing Tilt Angles: 0 -> ' + str(i+1) + ' / ' + str(Nproj))

    # Reset Beta.
    beta = beta0

    dd_vec = np.zeros(Niter_est*2, dtype=np.float32)
    tv_vec = np.zeros(Niter_est*2, dtype=np.float32)

    t0 = time.time()
 
    #Main Reconstruction Loop
    while True: 

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
        if (Niter[i] == 0):
            dPOCS = tomo_obj.matrix_2norm() * alpha
            dp = dPOCS / alpha
        else: # Measure change from ART.
            dp = tomo_obj.matrix_2norm() 

        # Measure difference between exp/sim projections.
        dd_vec[Niter[i]] = tomo_obj.dyn_vector_2norm(i+1)

        #Measure TV. 
        tv_vec[Niter[i]] = tomo_obj.tv()

        tomo_obj.copy_recon() 

        #TV Minimization. 
        tomo_obj.tv_gd(ng, dPOCS)
        dg = tomo_obj.matrix_2norm()

        if (dg > dp * r_max and dd_vec[Niter[i]] > eps):
            dPOCS *= alpha_red

        Niter[i] += 1

        #Calculate current time. 
        ctime = ( time.time() - t0 ) 

        if ctime > time_limit:
            break

    Niter_est = Niter[i]
    print('Number of Iterations: ' + str(Niter[i]) + '\n')

    #Remove Excess elements.
    dd_vec = dd_vec[:Niter[i]]
    tv_vec = tv_vec[:Niter[i]]

    #Append to final vector. 
    fdd_vec = np.append(fdd_vec, dd_vec)
    ftv_vec = np.append(ftv_vec, tv_vec)

    if save and (i+1)%5 == 0 :
        os.makedirs('Results/'+ file_name +'_Time/', exist_ok=True)
        recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
        for s in range(Nslice):
            recon[s,:,:] = tomo_obj.getRecon(s)
        np.save('Results/'+ file_name +'_Time/proj_' + str(i+1) + '_recon.npy', recon)

    if show_live_plot and (i+1) % 5 == 0:
        pr.time_tv_live_plot(fdd_vec,eps,ftv_vec,Niter,i)

#Save all the results to single matrix.
results = np.array([Niter, fdd_vec, eps, ftv_vec])

# Save the Reconstruction.
np.save('Results/'+ file_name +'_Time/final_recon.npy', recon)
np.save('Results/'+ file_name +'_Time/results.npy', results)

if show_final_plot:
    pr.time_results(fdd_vec, eps, ftv_vec, Niter)
