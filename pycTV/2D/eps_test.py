# Script to test the effect of TV reconstructions under multiple
# epsilon (Data Tolerance) values. 

import sys, os
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative, timer
from skimage import io
import numpy as np
import ctvlib 
import time
############ Parameters ################

# Number of Iterations (Main Loop)
Niter = 100

# Number of Iterations (TV Loop)
ng = 20

# Step Size for Theta
dTheta = 2

#ART Parameter
beta0 = 1.0

# ART Reduction.
beta_red = 0.995

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95

#TV Parameter
alpha = 0.5

#Minimum and Maximum Epsilon Values
min_eps = 0.1
max_eps = 3.1

#########################################

#Read Image. 
tiltSeries = io.imread('Test_Image/Co2P_256.tif')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
(Nx, Ny) = tiltSeries.shape
tiltSeries = tiltSeries.flatten()

# Generate Tilt Angles.
Nproj = 180/dTheta + 1
tiltAngles = np.linspace(0, 180, Nproj, dtype=np.float32)

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(int(Ny*Nproj), Ny*Ny)

# Generate measurement matrix
A = obj.parallelRay(Ny, tiltAngles)
obj.rowInnerProduct()

# Construct Reconstruction and Projections. 
b = np.transpose(A.dot(tiltSeries))

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
    recon = np.zeros([Nx, Ny], dtype=np.float32)
    tv_vec = np.zeros(Niter, dtype=np.float32)
    dd_vec = np.zeros(Niter, dtype=np.float32)

    #Main Loop
    for i in range(Niter): 

        temp_recon = recon.copy()
        
        obj.ART(np.ravel(recon), b, beta)  

        #Positivity constraint 
        recon[recon < 0] = 0  

        #ART-Beta Reduction
        beta = beta*beta_red 

        g = A.dot(np.ravel(recon))

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

    # Save the Data.
    os.makedirs('Results/Epsilon_Test/' + str(eps[k]), exist_ok=True)
    np.save('Results/Epsilon_Test/' + str(eps[k]) + '/tv.npy', tv_vec)
    np.save('Results/Epsilon_Test/' + str(eps[k]) + '/dd.npy', dd_vec)
    io.imsave('Results/Epsilon_Test/' + str(eps[k]) + '/slice.tif', recon/np.amax(recon))

    