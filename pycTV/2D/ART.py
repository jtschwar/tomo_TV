# General 2D - ART Reconstruction with Positivity Constraint.

import sys, os
sys.path.append('./Utils')
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import ctvlib 

Niter = 100
num_tilts = 30
beta = 1.0
beta_red = 0.95

#Read Image. 
tiltSeries = io.imread('Test_Image/Co2P_256.tif')
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

#Main Loop
for i in range(Niter): 

    if (i % 10 == 0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    # obj.recon(np.ravel(recon), b, beta)  
    obj.ART(recon.ravel(), b, beta)

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta = beta*beta_red 

# Display the Reconstruction. 
plt.imshow(recon,cmap='gray')
plt.axis('off')
plt.show()