from pytvlib import tv, tv_derivative
from skimage import io
import numpy as np
import ctvlib 
########################################

# Number of Iterations (Main Loop)
Niter = 10

# Number of Iterations (TV Loop)
ng = 5

# Parameter in ART Reconstruction.
beta = 1.0

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 1.0

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

##########################################

#Read Image. 
tiltSeries = io.imread('Co2P_tiltser.tiff')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
(Nproj, Nray, Nslice) = tiltSeries.shape 
b = np.zeros([Nray*Nproj, Nslice])
g = np.zeros([Nray*Nproj, Nslice])

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

for s in range(Nslice):
    b[:,s] = tiltSeries[:,:,s].ravel()
obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.linspace(-75, 75, 76, dtype=np.float32)

# Generate measurement matrix
obj.parallelRay(Nray, tiltAngles)
obj.rowInnerProduct()

recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32)

#Main Loop
for i in range(Niter): 

    print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    temp_recon = recon.copy()

    for s in range(Nslice):
        recon[:,:,s] = obj.recon(recon[:,:,s].ravel(), beta, s, Nray) 

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta = beta*beta_red 

    for s in range(Nslice):
        g[:,s] = obj.forwardProjection(recon[:,:,s].ravel())

    if (i == 0):
        dPOCS = np.linalg.norm(recon - temp_recon) * alpha

    dd = np.linalg.norm(g - b)
    dp = np.linalg.norm(recon - temp_recon)   
    temp_recon = recon.copy()

    for j in range(ng):
        v = tv_derivative(recon)
        v /= np.linalg.norm(v)
        recon -= dPOCS * v

    dg = np.linalg.norm(recon - temp_recon) 

    if (dg > dp * r_max and dd > eps):
        dPOCS *= alpha_red

# Save the Reconstruction.
np.save('Results/Co2P_recon.npy', recon)