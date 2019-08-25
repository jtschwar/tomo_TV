# General 2D - ASD - POCS (TV) Reconstruction Algorithm. 

import sys, os
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import ctvlib 

########################################

# Number of Iterations (Main Loop)
Niter = 500

# Number of Iterations (TV Loop)
ng = 10

# Step Size for Theta
dTheta = 2

# Parameter in ART Reconstruction.
beta = 1.0

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 5

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

##########################################

#Read Image. 
tiltSeries = io.imread('Test_Image/Co2P_256.tif')
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

#Main Loop
for i in range(Niter): 

    temp_recon = recon.copy()

    if (i % 100 == 0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    obj.ART(np.ravel(recon), b, beta)    

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta = beta*beta_red 

    g = A.dot(np.ravel(recon))

    if (i == 0):
        dPOCS = np.linalg.norm(recon - temp_recon) * alpha

    dd = np.linalg.norm(g - b) / g.size
    dd_vec[i] = dd
    dp = np.linalg.norm(recon - temp_recon)   
    temp_recon = recon.copy()

    for j in range(ng):
        v = tv_derivative(recon)
        v /= np.linalg.norm(v)
        recon -= dPOCS * v

    dg = np.linalg.norm(recon - temp_recon) 

    if (dg > dp * r_max and dd > eps):
        dPOCS *= alpha_red

    tv_vec[i] = tv(recon)
    rmse_vec[i] = np.sqrt(((recon - img0)**2).mean())

x = np.arange(tv_vec.shape[0]) + 1

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(5,4))
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

# # Display the Reconstruction. 
plt.figure()
plt.imshow(recon,cmap='gray')
plt.axis('off')
plt.show()