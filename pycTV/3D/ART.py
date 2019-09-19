# General 3D - ART Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
from pytvlib import parallelRay, timer, load_data
import plot_results as pr
import numpy as np
import ctvlib 
import time
########################################

file_name = '256_Co2P_tiltser.tif'

# Number of Iterations (Main Loop)
Niter = 100

# Parameter in ART Reconstruction.
beta = 1.0

# ART Reduction.
beta_red = 0.995

save = True #Save Final Reconstruction
show_fplot = True #Calculate dd and show final plot.
show_live_plot = True #Calculate dd and show intermediate plots.

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

dd_vec = np.zeros(Niter)

t0 = time.time()
counter = 1

#Main Loop
for i in range(Niter): 

    tomo_obj.ART(beta, -1)

    #Positivity constraint 
    tomo_obj.positivity()

    #ART-Beta Reduction
    beta *= beta_red 

    if show_plot or show_live_plot:
        tomo_obj.forwardProjection(-1)
        dd_vec[i] = tomo_obj.vector_2norm()

    if (i+1)%10 ==0:
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))
        timer(t0, counter, Niter)
    if 
        pr.live_ART_results(dd_vec,i)

    counter += 1

recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
for s in range(Nslice):
    recon[s,:,:] = tomo_obj.getRecon(s)

if save:
    np.save('Results/ART_'+file_name+'_recon.npy', recon)

if show_plot:
    pr.ART_results(dd_vec)
