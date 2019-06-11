# Script to test the effect of TV reconstructions under multiple
# epsilon (Data Tolerance) values. 

import sys, os
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative, parallelRay, timer
from skimage import io
import numpy as np
import ctvlib 
import time
########################################

# Number of Iterations (Main Loop)
Niter = 50

# Number of Iterations (TV Loop)
ng = 5

# ART Reduction.
beta_red = 0.995

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95

#TV Parameter
alpha = 0.5

#ART Parameter
beta0 = 1.0

#Minimum and Maximum Epsilon Values
min_eps = 0.1
max_eps = 2.1

##########################################

#Read Image. 
tiltSeries = io.imread('Tilt_Series/Co2P_tiltser.tiff')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
tiltSeries = np.swapaxes(tiltSeries, 0, 2)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros( [Nslice, Nray*Nproj] )
g = np.zeros([Nslice, Nray*Nproj])

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

for s in range(Nslice):
    b[s,:] = tiltSeries[s,:,:].transpose().ravel()
obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/Co2P_tiltAngles.npy')

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
obj.create_measurement_matrix(A)
A = None
obj.rowInnerProduct()

#Array of Data Tolerance Parameters. 
eps = np.arange(min_eps, max_eps, 0.1)
eps = np.around(eps, decimals=1)

t0 = time.time()
counter = 1

for k in range(len(eps)):

    print('Reconstructing with Epsilon = ' + str(eps[k]))

    # Reset Beta.
    beta = beta0

    #Create Vectors.
    recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32)
    tv_vec = np.zeros(Niter, dtype=np.float32)
    dd_vec = np.zeros(Niter, dtype=np.float32)

    #Main Loop
    for i in range(Niter): 

        temp_recon = recon.copy()

        for s in range(Nslice):
            recon[s,:,:] = obj.ART(recon[s,:,:].ravel(), beta, s, -1) 

        #Positivity constraint 
        recon[recon < 0] = 0  

        #ART-Beta Reduction
        beta = beta*beta_red 

        for s in range(Nslice):
            g[s,:] = obj.forwardProjection(recon[s,:,:].ravel(), -1)

        if (i == 0):
            dPOCS = np.linalg.norm(recon - temp_recon) * alpha

        dd_vec[i] = np.linalg.norm(g - b) / g.size
        dp = np.linalg.norm(recon - temp_recon)   
        temp_recon = recon.copy()

        for j in range(ng):
            v = tv_derivative(recon)
            v /= np.linalg.norm(v)
            recon -= dPOCS * v
        tv_vec[i] = tv(recon)

        dg = np.linalg.norm(recon - temp_recon) 

        if (dg > dp * r_max and dd_vec[i] > eps[k]):
            dPOCS *= alpha_red

    timer(t0, counter, len(eps))
    counter += 1

    # Save the Reconstruction.
    os.makedirs('Results/Epsilon_Test/' + str(eps[k]), exist_ok=True)
    np.save('Results/Epsilon_Test/' + str(eps[k]) + '/recon.npy', recon)
    np.save('Results/Epsilon_Test/' + str(eps[k]) + '/tv.npy', tv_vec)
    np.save('Results/Epsilon_Test/' + str(eps[k]) + '/dd.npy', dd_vec)

    im = recon[:,134,:]/np.amax(recon[:,134,:])
    io.imsave('Results/Epsilon_Test/' + str(eps[k]) + '/slice.tif', im)

    