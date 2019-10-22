# General 3D - SIRT Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
from pytvlib import parallelRay, timer, load_data
import numpy as np
import ctvlib 
import time
########################################

file_name = '256_Co2P_tiltser.tif'

# Number of Iterations (Main Loop)
Niter = 20

# Parameter in SIRT Reconstruction.
beta = 0.0001

# ART Reduction.
beta_red = 0.995

# Save Final Reconstruction. 
save = True

##########################################

# #Read Image. 
(file_name, tiltSeries) = load_data(file_name)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros([Nslice, Nray*Nproj])

# Initialize C++ Object.. 
tomo_obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

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
tomo_obj.initialize_SIRT()

dd_vec = np.zeros(Niter)

t0 = time.time()
counter = 1

#Main Loop
for i in range(Niter): 

    if (i%10 ==0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    tomo_obj.SIRT(beta, -1)

    #Positivity constraint 
    tomo_obj.positivity()

    #ART-Beta Reduction
    beta *= beta_red 

    if (i%10 ==0):
        timer(t0, counter, Niter)
    counter += 1

recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
for s in range(Nslice):
    recon[s,:,:] = tomo_obj.getRecon(s)

if save:
    np.save('Results/SIRT_'+file_name+'_recon.npy', recon)
