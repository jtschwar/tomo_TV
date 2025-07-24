# General 2D - ASD - POCS (TV) Reconstruction Algorithm. 

import sys, os
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import ctvlib 
import cv2

########################################

# Name of input file. 
file_name = 'Co2P.tif'

# Number of Iterations (Main Loop)
Niter = 1000

# Number of Iterations (TV Loop)
ng = 20

# Step Size for Theta
dTheta = 1

# ART Parameter.
beta = 1.0

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 0

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 1.0

# Save and show reconstruction. 
save = False
show = True

##########################################

#Read Image. 
tiltSeries = io.imread('Test_Image/'  + file_name)
tiltSeries = np.array(tiltSeries, dtype=np.float32)
tiltSeries /= np.amax(tiltSeries)
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
A = obj.parallelRay(Ny, tiltAngles)
obj.rowInnerProduct()

b = np.transpose(A.dot(tiltSeries))
recon = np.zeros([Nx, Ny], dtype=np.float32)
dd_vec = np.zeros(Niter)
tv_vec = np.zeros(Niter)
rmse_vec = np.zeros(Niter)
cos_alph_vec = np.zeros(Niter)

#Main Loop
for i in range(Niter): 

    temp_recon = recon.copy()

    if (i % 100 == 0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    obj.ART(np.ravel(recon), b, beta)    

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta *= beta_red 

    g = A.dot(np.ravel(recon))

    if (i == 0):
        dPOCS = np.linalg.norm(recon - temp_recon) * alpha

    dd_vec[i]= np.linalg.norm(g - b) / g.size
    dp = np.linalg.norm(recon - temp_recon)   
    temp_recon = recon.copy()

    recon[:] = obj.tv_loop(recon, dPOCS, ng)

    dg = np.linalg.norm(recon - temp_recon) 

    if (dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

    tv_vec[i] = tv(recon)
    rmse_vec[i] = np.sqrt(((recon - img0)**2).mean())
    cos_alph_vec[i] = obj.CosAlpha(recon, b, g, int(Ny*Nproj))

if save:
    #Save all the results to single matrix.
    results = np.array([tv_vec, dd_vec, rmse_vec, cos_alph_vec])
    np.save('Results/results.npy', results)

    # Save Image.
    recon[ recon < 0] = 0
    io.imsave('Results/TV_Recon_' + file_name, np.uint16(recon*255))
    

if show:
    x = np.arange(tv_vec.shape[0]) + 1

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(7,6))
    fig.subplots_adjust(hspace=0.4)

    ax1.plot(x, tv_vec,color='blue', linewidth=2.0)
    ax1.set_title('Final TV: ' +str(tv_vec[-1]), loc='right', fontsize=10)
    ax1.set_title('TV', loc='center', fontweight='bold')
    ax1.axhline(y=tv0, color='r')
    ax1.set_xticklabels([])

    ax2.plot(x,dd_vec,color='black', linewidth=2.0)
    ax2.axhline(y=eps, color='r')
    ax2.set_title('Final dd: ' +str(dd_vec[-1]), loc='right', fontsize=10)
    ax2.set_title('DD', loc='left', fontweight='bold')
    ax2.set_xticklabels([])

    ax3.plot(x, rmse_vec, color='m', linewidth=2.0)
    ax3.set_title('Final RMSE: ' +str(rmse_vec[-1]), loc='right', fontsize=10)
    ax3.set_title('RMSE', loc='left', fontweight='bold')
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

    plt.figure()
    plt.plot(x, cos_alph_vec)
    plt.xlabel('Number of Iterations')
    plt.label('Cosine-Alpha', loc='left', fontweight='bold')
    plt.show()