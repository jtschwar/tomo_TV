# Script to simulate tomography reconstructions when a new projection 
# is added every three minutes.

import sys, os
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative, parallelRay
from skimage.io import imread, imsave
from matplotlib import pyplot as plt
import scipy.sparse as ss
import numpy as np
import ctvlib
import time

########################################

# Name of input file. 
file_name = 'Co2P.tif'

# Number of Iterations (TV Loop)
ng = 20

# Step Size for Theta
dTheta = 2

# ART Parameter.
beta0 = 1.0

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 5

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

#Amount of time before next projection is collected (Seconds).
time_limit = 180 # Seconds
Ncores = 1

# Save and show reconstruction. 
save = True
show = False

##########################################

#Read Image. 
tiltSeries = imread('Test_Image/' + file_name)
tiltSeries = np.array(tiltSeries, dtype=np.float32)
tv0 = tv(tiltSeries)
img0 = tiltSeries.copy()
(Nx, Ny) = tiltSeries.shape
tiltSeries = tiltSeries.flatten()

# Generate Tilt Angles.
Nproj = 180/dTheta + 1
tiltAngles = np.linspace(0, 180, int(Nproj), dtype=np.float32)

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(int(Ny*Nproj), Ny*Ny)

# Generate measurement matrix
pyA = parallelRay(Ny, tiltAngles)
obj.create_measurement_matrix(pyA)
A = ss.coo_matrix((pyA[0,:], (pyA[1,:],pyA[2,:])), shape = (Ny*Nproj, Ny**2), dtype = np.float32  )
obj.rowInnerProduct()

#Generate Reconstruction. 
b = np.transpose(A.dot(tiltSeries))
recon = np.zeros([Nx, Ny], dtype=np.float32)
Niter = np.zeros(int(Nproj), dtype=np.int32)

#Final vectors for dd, tv, and rmse. 
results = np.array([], dtype=np.float32)
fdd_vec = np.array([])
ftv_vec = np.array([])
frmse_vec = np.array([])
Niter_est = 10000

# time_limit  = time_limit / Nslice * Ncores

#Dynamic Tilt Series Loop. 
for i in range(int(Nproj)):

    print('Reconstructing Tilt Angles: 0 -> ' + str(i+1) + '/' + str(int(Nproj)))

    dd_vec = np.zeros(Niter_est, dtype=np.float32)
    tv_vec = np.zeros(Niter_est, dtype=np.float32)
    rmse_vec = np.zeros(Niter_est, dtype=np.float32)
    cos_alph_vec = np.zeros(Niter_est, dtype=np.float32)

    # Reset Beta.
    beta = beta0

    # Keep track of time. 
    t0 = time.time()

    max_row = int(Nx * (i + 1))
 
    #Main Reconstruction Loop
    while True: 

        temp_recon = recon.copy()

        #ART Reconstruction. 
        obj.ART2(np.ravel(recon), b, beta, int(Nx*(i+1))) 

        #Positivity constraint 
        recon[recon < 0] = 0 

        #ART-Beta Reduction.
        beta *= beta_red

        #Forward Projection
        g = A[:Nx*(i+1)].dot(np.ravel(recon))

        # #Measure cosine-alpha
        cos_alph_vec[Niter[i]] = obj.CosAlpha(recon, b, g, max_row)

        if (i == 0):
            dPOCS = np.linalg.norm(recon - temp_recon) * alpha

        dd_vec[Niter[i]] = np.linalg.norm(g - b[:max_row]) / g.size
        dp = np.linalg.norm(recon - temp_recon)   
        temp_recon = recon.copy()

        recon[:] = obj.tv_loop(recon, dPOCS, ng)

        dg = np.linalg.norm(recon - temp_recon) 

        if (dg > dp * r_max and dd_vec[Niter[i]] > eps):
            dPOCS *= alpha_red

        tv_vec[Niter[i]] = tv(recon)
        Niter[i] += 1

        #Calculate current time. 
        ctime = ( time.time() - t0 ) 

        if ctime > time_limit:
            break

    Niter_est = Niter[i]
    print('Number of Iterations: ' + str(Niter[i]) + '\n')

    #Remove Excess elements
    tv_vec = tv_vec[:Niter[i]+1]
    ftv_vec = np.append(ftv_vec, tv_vec)
    dd_vec = dd_vec[:Niter[i]+1]
    fdd_vec = np.append(fdd_vec, dd_vec)
    rmse_vec = rmse_vec[:Niter[i]+1]
    frmse_vec = np.append(frmse_vec, rmse_vec) 

    if save:
        recon[ recon < 0] = 0
        imsave('Results/Time/Proj_' + str(i) + '.tif', np.uint16(recon))


if save:
    # Save Data. 
    os.makedirs('Results/Time/' + str(i+1), exist_ok=True)
    np.save('Results/Time/tv.npy', ftv_vec)
    np.save('Results/Time/dd.npy', fdd_vec)
    np.save('Results/Time/rmse.npy', frmse_vec)
    np.save('Results/Time/Niter.npy', Niter)
    imsave('Final_Recon.tif', recon)

if show:

    x = np.arange(tv_vec.shape[0]) + 1

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(7,6))
    fig.subplots_adjust(hspace=0.4)

    ax1.plot(x, tv_vec,color='blue', linewidth=2.0)
    ax1.set_title('Min TV: ' +str(np.amin(tv_vec)), loc='right', fontsize=10)
    ax1.set_title('TV', loc='center', fontweight='bold')
    ax1.axhline(y=tv0, color='r')
    ax1.set_xticklabels([])

    ax2.plot(x,dd_vec,color='black', linewidth=2.0)
    ax2.axhline(y=eps, color='r')
    ax2.set_title('Min dd: ' +str(dd_vec[-1]), loc='right', fontsize=10)
    ax2.set_title('DD', loc='left', fontweight='bold')
    ax2.set_xticklabels([])

    ax3.plot(x, rmse_vec, color='m', linewidth=2.0)
    ax3.set_title('Min RMSE: ' +str(rmse_vec[-1]), loc='right', fontsize=10)
    ax2.set_title('RMSE', loc='left', fontweight='bold')
    ax3.set_xlabel('Number of Iterations', fontweight='bold')

    if save: 
        file_name = file_name.replace('.tif', '')
        plt.savefig('time_plot_' + file_name + '.png')

    # # Display the Reconstruction. 
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,3))
    ax1.imshow(img0,cmap='gray')
    ax1.axis('off')
    ax1.set_title('Original Image')
    ax2.imshow(recon,cmap='gray')
    ax2.axis('off')
    ax2.set_title('Reconstruction')
    plt.show()

