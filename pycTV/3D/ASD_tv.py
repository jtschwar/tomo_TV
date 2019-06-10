import sys
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

# Parameter in ART Reconstruction.
beta = 1.0

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 0.2

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.5

##########################################

# #Read Image. 
tiltSeries = io.imread('Tilt_Series/Co2P_tiltser.tiff')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
tiltSeries = np.swapaxes(tiltSeries, 0, 2)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros([Nslice, Nray*Nproj])
g = np.zeros([Nslice, Nray*Nproj])

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

for s in range(Nslice):
    b[s,:] = tiltSeries[s,:,:].transpose().ravel()
obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.linspace(-75, 75, 76, dtype=np.float32)

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

    dd = np.linalg.norm(g - b) / g.size
    dp = np.linalg.norm(recon - temp_recon)   
    temp_recon = recon.copy()

    for j in range(ng):
        v = tv_derivative(recon)
        v /= np.linalg.norm(v)
        recon -= dPOCS * v
    recon[recon < 0] = 0

    dg = np.linalg.norm(recon - temp_recon) 

    if (dg > dp * r_max and dd > eps):
        dPOCS *= alpha_red

    timer(t0, counter, Niter)
    counter += 1

# Save the Reconstruction.
# np.save('Results/Co2P_recon.npy', recon)

im = recon[:,134,:]/np.amax(recon[:,134,:])
io.imsave('Results//slice.tif', im)
