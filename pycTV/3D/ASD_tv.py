import sys
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative, parallelRay, timer
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import ctvlib 
import time
########################################

# Number of Iterations (Main Loop)
Niter = 200

# Number of Iterations (TV Loop)
ng = 10

# Parameter in ART Reconstruction.
beta = 1.0

# ART Reduction.
beta_red = 0.975

# Data Tolerance Parameter
eps = 0.00146 

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

##########################################

#Read Image. 
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
tiltAngles = np.load('Tilt_Series/Co2P_tiltAngles.npy')

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
obj.create_measurement_matrix(A)
A = None
obj.rowInnerProduct()

recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32)
dd_vec = np.zeros(Niter)
tv_vec = np.zeros(Niter)
gradient_vec = np.zeros(Niter)

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

    if (i == Niter - 1):
        np.save('Results/FePt_Recon.npy', recon)

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
    recon[recon < 0] = 0

    dg = np.linalg.norm(recon - temp_recon) 

    if (dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

    tv_vec[i] = tv(recon)


    timer(t0, counter, Niter)
    counter += 1

x = np.arange(tv_vec.shape[0]) + 1

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,4))
fig.subplots_adjust(hspace=0.4)

ax1.plot(x, tv_vec,color='blue', linewidth=2.0)
ax1.set_title('Min TV: ' +str(np.amin(tv_vec)), loc='right', fontsize=10)
ax1.set_title('TV', loc='center', fontweight='bold')
ax1.set_xticklabels([])

ax2.plot(x,dd_vec,color='black', linewidth=2.0)
ax2.axhline(y=eps, color='r')
ax2.set_title('Min dd: ' +str(dd_vec[-1]), loc='right', fontsize=10)
ax2.set_title('DD', loc='center', fontweight='bold')
ax2.set_xlabel('Number of Iterations', fontweight='bold')
