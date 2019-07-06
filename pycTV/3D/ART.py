# General 3D - ART Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
from pytvlib import parallelRay, timer
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import ctvlib 
import time
########################################

# Number of Iterations (Main Loop)
Niter = 200

# Parameter in ART Reconstruction.
beta = 1.0

# ART Reduction.
beta_red = 0.995

##########################################

# #Read Image. 
# tiltSeries = io.imread('Tilt_Series/Co2P_tiltser.tiff')
tiltSeries = np.load('Tilt_Series/FePt_projections.npy')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
# tiltSeries = np.swapaxes(tiltSeries, 0, 2)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros([Nslice, Nray*Nproj], dtype=np.float32)
g = np.zeros([Nslice, Nray*Nproj], dtype=np.float32)

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

for s in range(Nslice):
    b[s,:] = tiltSeries[s,:,:].transpose().ravel()
obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/FePt_tiltAngles.npy')

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
obj.create_measurement_matrix(A)
A = None
obj.rowInnerProduct()

recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F')
dd_vec = np.zeros(Niter)

t0 = time.time()
counter = 1

#Main Loop
for i in range(Niter): 

    print('Iteration No.: ' + str(i+1) +'/'+str(Niter))


    for s in range(Nslice):
        recon[s,:,:] = obj.ART(recon[s,:,:].flatten(), beta, s, -1) 

    for s in range(Nslice):
        g[s,:] = obj.forwardProjection(recon[s,:,:].ravel(), -1)

    dd_vec[i] = np.linalg.norm(g - b) / g.size

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta *= beta_red 

    timer(t0, counter, Niter)
    counter += 1

x = np.arange(dd_vec.shape[0]) + 1 
plt.plot(x,dd_vec,color='black', linewidth=2.0)
plt.title('Last dd: ' +str(dd_vec[i]), loc='right', fontsize=10)
plt.title('DD', loc='left', fontweight='bold')
plt.xlabel('Number of Iterations', fontweight='bold')
plt.show()
