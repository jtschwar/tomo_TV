# General 3D - ART Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
import plot_results as pr
from pytvlib import *
import numpy as np
import ctvlib 
import time
########################################

vol_size = '256_'
file_name = 'Co2P_tiltser.tif'

# Number of Iterations (Main Loop)
Niter = 100

# Parameter in ART Reconstruction.
beta0 = 1.0

# ART Reduction.
beta_red = 0.995

saveRecon = True #Save Final Reconstruction
show_live_plot = False #Calculate dd and show intermediate plots.

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

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
tomo_obj.load_A(A)
A = None
tomo_obj.rowInnerProduct()

dd_vec = np.zeros(Niter)

beta = beta0
t0 = time.time()
counter = 1

#Main Loop
for i in range(Niter): 

    tomo_obj.ART(beta, -1)

    #Positivity constraint 
    tomo_obj.positivity()

    #ART-Beta Reduction
    beta *= beta_red 

    tomo_obj.forwardProjection(-1)
    dd_vec[i] = tomo_obj.vector_2norm()

    if (i+1) % 25 ==0:
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))
        timer(t0, counter, Niter)
        if show_live_plot:
            pr.live_ART_results(dd_vec,i)

    counter += 1

print('Reconstruction Complete, Saving Data..')
print('Save Recon :: {}'.format(saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_ART'
meta = {'Niter':Niter,'beta':beta0,'beta_red':beta_red}
results = {'dd':dd_vec}
save_results([fDir,fName], meta, results)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)
