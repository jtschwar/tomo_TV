# General 3D - SIRT Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
from pytvlib import *
import numpy as np
import ctvlib 
import time
########################################

vol_size = '256_'
file_name = 'Co2P_tiltser.npy'

# Number of Iterations (Main Loop)
Niter = 100

# Save Final Reconstruction. 
saveRecon = True

##########################################

# #Read Image. 
(fName, tiltSeries) = load_data(vol_size,file_name)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros([Nslice, Nray*Nproj])

# Initialize C++ Object.. 
tomo_obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

for s in range(Nslice):
    b[s,:] = tiltSeries[s,:,:].transpose().ravel()
tomo_obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+fName+'_tiltAngles.npy')
Nproj = tiltAngles.shape[0]

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
tomo_obj.load_A(A)
A = None
beta = 1/tomo_obj.lipschits()
print('Measurement Matrix is Constructed!')

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

    #Forward Projection.
    tomo_obj.forwardProjection(-1)

    # Measure Data Distance (DD)
    dd_vec[i] = tomo_obj.vector_2norm()

    if (i%10 ==0):
        timer(t0, counter, Niter)
    counter += 1

print('Reconstruction Complete, Saving Data..')
print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_SIRT'
meta = {'vol_size':vol_size,'Niter':Niter,'beta':beta}
results = {'rmse':dd_vec}
save_results([fDir,fName], meta, results)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)
