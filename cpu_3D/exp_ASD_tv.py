# General 3D - ASD/TV Reconstruction with Positivity Constraint. 
# Intended for experimental datasets.
# RMSE and Volume's Original TV is unknown. 

import sys
sys.path.append('./Utils')
import plot_results as pr
from pytvlib import *
import numpy as np
import ctvlib 
import time
########################################

vol_size = '256_'
file_name = '180_tiltser.tif'

# Number of Iterations (Main Loop)
Niter = 100

# Number of Iterations (TV Loop)
ng = 10

# Parameter in ART Reconstruction.
beta0 =  0.25

# ART Reduction.
beta_red = 0.99

# Data Tolerance Parameter
eps = 0.002

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

#Outcomes:
show_live_plot = 0
saveGif, saveRecon = True, True           
gif_slice = 156
##########################################

# Read Image. 
(fName, tiltSeries) = load_data(vol_size,fName)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros([Nslice, Nray*Nproj])

# Initialize C++ Object.. 
tomo_obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

# Set Tilt Series. 
for s in range(Nslice):
    b[s,:] = tiltSeries[s,:,:].transpose().ravel()
tomo_obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+ fName +'_tiltAngles.npy')

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
tomo_obj.load_A(A)
A = None
tomo_obj.rowInnerProduct()
print('Measurement Matrix is Constructed!')

# Initialize Vectors for DD, TV, and Time Elapsed. 
dd_vec, tv_vec, time_vec = np.zeros(Niter), np.zeros(Niter), np.zeros(Niter)
beta = beta0
counter = 1 
t0 = time.time()

#Main Loop
for i in range(Niter): 

    if ( i % 25 ==0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    tomo_obj.copy_recon()

    #ART Reconstruction. 
    tomo_obj.sART(beta, -1)

    #Positivity constraint 
    tomo_obj.positivity()

    #ART-Beta Reduction
    beta *= beta_red 

    #Forward Projection.
    tomo_obj.forwardProjection(-1)

    #Measure Magnitude for TV - GD.
    if (i == 0):
        dPOCS = tomo_obj.matrix_2norm() * alpha
        dp = dPOCS / alpha
    else: # Measure change from ART.
        dp = tomo_obj.matrix_2norm() 

    # Measure difference between exp/sim projections.
    dd_vec[i] = tomo_obj.vector_2norm()

    #Measure TV. 
    tv_vec[i] = tomo_obj.tv()

    tomo_obj.copy_recon() 

    #TV Minimization. 
    tomo_obj.tv_gd(ng, dPOCS)
    dg = tomo_obj.matrix_2norm()

    if(dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

    if ((i+1) % 15 ==0):
        timer(t0, counter, Niter)
        if show_live_plot:
            pr.exp_ASD_live_plot(dd_vec, eps, tv_vec, i)

    counter += 1

print('Reconstruction Complete, Saving Data..')
print('Save Gif :: {}, Save Recon :: {}'.format(saveGif, saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_ASD'
meta = {'Niter':Niter,'ng':ng,'beta':beta0,'beta_red':beta_red,'eps':eps}
meta.update({'r_max':r_max,'alpha':alpha,'alpha_red':alpha_red,'vol_size':vol_size})
results = {'dd':dd_vec,'eps':eps,'tv':tv_vec,'time':time_vec}
save_results([fDir,fName], meta, results)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)