# General 2D - ART Reconstruction with Positivity Constraint.

import sys, os
sys.path.append('./Utils')
from matplotlib import pyplot as plt
import numpy as np
import ctvlib 
import cv2

###### Parameters ########

Niter = 300
num_tilts = 30
beta = 1.0
beta_red = 0.95

#########################

#Read Image. 
tiltSeries = cv.imread('Test_Image/Co2P_256.tif')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
(Nx, Ny) = tiltSeries.shape
tiltSeries = tiltSeries.flatten()

# Generate Tilt Angles.
tiltAngles = np.linspace(0, 180, num_tilts, dtype=np.float32)

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(Ny*num_tilts, Ny*Ny)

# Generate measurement matrix
A = obj.parallelRay(Ny, tiltAngles)
obj.rowInnerProduct()

b = np.transpose(A.dot(tiltSeries))
recon = np.zeros([Nx, Ny], dtype=np.float32)
dd_vec = np.zeros(Niter)

#Main Loop
for i in range(Niter): 

    if (i % 10 == 0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))
 
    obj.ART(recon.ravel(), b, beta)

    #Positivity constraint 
    recon[recon < 0] = 0 

    g = A.dot(np.ravel(recon))
    dd_vec[i] = np.linalg.norm(g - b) / g.size 

    #ART-Beta Reduction
    beta = beta*beta_red 


x = np.arange(dd_vec.shape[0]) + 1
plt.plot(x,dd_vec)
plt.xlabel('Number of Iterations')
plt.title('DD', fontweight='bold')

# # Display the Reconstruction. 
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,3))
ax1.imshow(tiltSeries.reshape(Nx,Ny),cmap='gray')
ax1.axis('off')
ax1.set_title('Original Image')
ax2.imshow(recon,cmap='gray')
ax2.axis('off')
ax2.set_title('Reconstruction')
plt.show()