import sys, os
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative 
from skimage.io import imread
import numpy as np
import ctvlib
import time
########################################

# Number of Iterations (TV Loop)
ng = 10

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 1.0

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.5

time_limit = 180

##########################################

#Read Image. 
tiltSeries = imread('Tilt_Series/Co2P_tiltser.tiff')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
(Nproj, Nray, Nslice) = tiltSeries.shape 
b = np.zeros([Nslice, Nray*Nproj])

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

#Transfer Tilt Series to C++ Object. 
for s in range(Nslice):
    b[s,:] = tiltSeries[:,:,s].ravel()
obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.linspace(-75, 75, 76, dtype=np.float32)

# Generate measurement matrix
obj.parallelRay(Nray, tiltAngles)
obj.rowInnerProduct()

#Generate Reconstruction. 
recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32)
Niter = np.zeros(Nproj)

#Dynamic Tilt Series Loop. 
for i in range(Nproj):

    print('Reconstructing Tilt Angles: 0 -> ' + str(i+1) )

    # Reset Beta.
    beta = 1.0

    tv_vec = np.zeros(Nproj)
    dd_vec = np.zeros(Nproj)
    g = np.zeros([Nslice, Nray * (i+1)])

    k = 0

    t0 = time.time()
 
    #Main Reconstruction Loop
    while True: 

        temp_recon = recon.copy()

        #ART Reconstruction. 
        for s in range(Nslice):
            recon[:,:,s] = obj.recon(recon[:,:,s].ravel(), beta, s, i+1) 

        #Positivity constraint 
        recon[recon < 0] = 0  

        #ART-Beta Reduction.
        beta = beta*beta_red 

        #Forward Projection/
        for s in range(Nslice):
            g[s,:] = obj.forwardProjection(recon[:,:,s].ravel(), i+1)

        if (i == 0):
            dPOCS = np.linalg.norm(recon - temp_recon) * alpha

        dd_vec[k] = np.linalg.norm(g - b[:,:Nray * (i+1)]) / g.size
        dp = np.linalg.norm(recon - temp_recon)   
        temp_recon = recon.copy()

        #TV Loop
        for j in range(ng):
            v = tv_derivative(recon)
            v /= np.linalg.norm(v)
            recon -= dPOCS * v

        dg = np.linalg.norm(recon - temp_recon) 

        if (dg > dp * r_max and dd_vec[k] > eps):
            dPOCS *= alpha_red

        tv_vec[k] = tv(recon)
        k += 1

        #Calculate current time. 
        ctime = ( time.time() - t0 ) 

        if ctime > time_limit:
            break

    Niter[i] = k
    print('Number of Iterations: ' + str(k) + '\n')

    # Save Data. 
    os.makedirs('Results/Time/' + str(i+1), exist_ok=True)
    np.save('Results/Time/' + str(i+1) + '/tv.npy', tv_vec)
    np.save('Results/Time/' + str(i+1) + '/dd.npy', dd_vec)

    if (i % 10 == 0):
        np.save('Results/Time/' + str(i+1) + '/recon.npy', recon)

# Save the Reconstruction.
np.save('Results/Time/Co2P_recon.npy', recon)
np.save('Results/Time/Niter.npy', Niter)