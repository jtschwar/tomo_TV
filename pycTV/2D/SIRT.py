# General 2D - ART Reconstruction with Positivity Constraint.

import sys, os
sys.path.append('./Utils')
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import ctvlib 

###### Parameters ########

Niter = 1000
num_tilts = 50
beta = 0.0001

show = True
#########################

#Read Image. 
tiltSeries = io.imread('Test_Image/phantom.tif')
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

    if (i % 100 == 0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    obj.SIRT(recon.ravel(), b, beta)

    beta *= 0.995

    #Positivity constraint 
    recon[recon < 0] = 0  


 # # Display the Reconstruction. 
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(7,3))
ax1.imshow(tiltSeries.reshape(Nx,Ny),cmap='gray')
ax1.axis('off')
ax1.set_title('Original Image')
ax2.imshow(recon,cmap='gray')
ax2.axis('off')
ax2.set_title('Reconstruction')
plt.show()