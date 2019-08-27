# General 2D - ASD - POCS (TV) Reconstruction Algorithm. 

import sys, os
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import ctvlib 
import cv2

########################################

file_name = 'Co2P.tif'

# Number of Iterations (Main Loop)
Niter = 1200

# Number of Iterations (TV Loop)
ng = 20

# Step Size for Theta
dTheta = 2

# Parameter in ART Reconstruction.
beta = 0.5

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 100

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

save = True

##########################################

#Read Image. 
tiltSeries = cv2.imread('Test_Image/' + file_name)
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
A = obj.parallelRay(Ny, tiltAngles)
obj.rowInnerProduct()

b = np.transpose(A.dot(tiltSeries))
recon = np.zeros([Nx, Ny], dtype=np.float32)
dd_vec = np.zeros(Niter)
tv_vec = np.zeros(Niter)
rmse_vec = np.zeros(Niter)
eps_vec = np.zeros([int(Niter//300), 2])
eps_ind = 0

recalc_l2 = False

#Main Loop
for i in range(Niter): 

    temp_recon = recon.copy()

    if (i % 500 == 0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    obj.ART(np.ravel(recon), b, beta)    

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta *= beta_red 

    g = A.dot(np.ravel(recon))

    if (i == 0):
        dPOCS = np.linalg.norm(recon - temp_recon) * alpha

    if recalc_l2:
        dPOCS = np.linalg.norm(recon - temp_recon) * alpha
        recalc_l2 = False

    dd_vec[i] = np.linalg.norm(g - b) / g.size
    dp = np.linalg.norm(recon - temp_recon)   
    temp_recon = recon.copy()

    for j in range(ng):
        v = tv_derivative(recon)
        v /= np.linalg.norm(v)
        recon -= dPOCS * v

    dg = np.linalg.norm(recon - temp_recon) 

    if (dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

    tv_vec[i] = tv(recon)
    rmse_vec[i] = np.sqrt( ((recon - img0)**2).mean() )

    if ((i+1) % 300 == 0 ):

        if save: #save image
            recon[recon<0] = 0
            cv2.imwrite('eps_' + str(round(eps,2)) + '_' + file_name, np.uint16(recon))

        # add new epsilon parameters
        eps_vec[eps_ind,:] = (eps, i)
        eps_ind += 1
        eps -= 5
        beta = 0.3
        recalc_l2 = True

file_name = file_name.replace('.tif', '')
x = np.arange(tv_vec.shape[0]) + 1

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(7,6))
fig.subplots_adjust(hspace=0.4)

ax1.plot(x, tv_vec,color='blue', linewidth=2.0)
ax1.set_title('Final TV: ' +str(np.amin(tv_vec)), loc='right', fontsize=10)
ax1.set_title('TV', loc='center', fontweight='bold')
ax1.set_ylim(bottom=np.amin(tv_vec)/2, top=tv_vec[-1]*2)
ax1.axhline(y=tv0, color='r')
ax1.set_xticklabels([])

ax2.plot(x,dd_vec,color='black', linewidth=2.0)
ax2.set_title('Final dd: ' +str(dd_vec[-1]), loc='right', fontsize=10)
ax2.set_title('DD', loc='left', fontweight='bold')
ax2.set_ylim(bottom=0, top=120)
ax2.set_xticklabels([])

ax3.plot(x, rmse_vec, color='m', linewidth=2.0)
ax3.set_title('Final RMSE: ' +str(rmse_vec[-1]), loc='right', fontsize=10)
ax3.set_title('RMSE', loc='left', fontweight='bold')
ax3.set_xlabel('Number of Iterations', fontweight='bold')

colors = itertools.cycle(sns.color_palette())
ax2.hlines(y=eps_vec[0,0], xmin=0, xmax=eps_vec[0,1], color=next(colors))
ax2.axvline(x=0, color='y', linestyle='dashed')
ax3.axvline(x=0, color='y', linestyle='dashed')
ax2.axvline(x=eps_vec[0,1], color='y', linestyle='dashed')
ax3.axvline(x=eps_vec[0,1], color='y', linestyle='dashed')
for i in range(1,eps_vec.shape[0]):
    ax2.hlines(y=eps_vec[i,0], xmin=eps_vec[i-1,1], xmax=eps_vec[i,1], color=next(colors))
    ax2.axvline(x=eps_vec[i,1], color='y', linestyle='dashed')
    ax3.axvline(x=eps_vec[i,1], color='y', linestyle='dashed')

if save:
    np.save('dyn_eps.npy', eps_vec)
    np.save('dyn_tv.npy', tv_vec)
    np.save('dyn_dd.npy', dd_vec)
    np.save('dyn_rmse.npy', rmse_vec)
    plt.savefig('dyn_eps_plot_' + file_name +'.png')

plt.show()

