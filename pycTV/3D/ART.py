# General 3D - ART Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
from skimage import io
from pytvlib import timer, parallelRay
import numpy as np
import ctvlib 
import time

Niter = 15
beta = 1.0
beta_red = 0.95

#Read Image. 
tiltSeries = io.imread('Tilt_Series/Co2P_tiltser.tiff')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
tiltSeries = np.swapaxes(tiltSeries, 0, 2)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros( [Nslice, Nray*Nproj] )

# Generate Tilt Angles.
tiltAngles = np.linspace(-75, 75, 76, dtype=np.float32)

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
obj.create_measurement_matrix(A)
A = None
obj.rowInnerProduct()

recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32)

t0 = time.time()
counter = 1

#Main Loop
for i in range(Niter): 

    print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    for s in range(Nslice):
        recon[s,:,:] = obj.ART(recon[s,:,:].ravel(), tiltSeries[s,:,:].transpose().ravel().astype(np.float32), beta)

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta = beta*beta_red 

    timer(t0, counter, Niter)
    counter += 1

# Save the Reconstruction.
np.save('Results/ART_recon.npy', recon)