from matplotlib import pyplot as plt
# import pathos.multiprocessing as mp
from skimage import io
import numpy as np
import ctvlib 

Niter = 10
beta = 1.0
beta_red = 0.95

#Read Image. 
tiltSeries = io.imread('Co2P_tiltser.tiff')
tiltSeries = np.array(tiltSeries, dtype=np.float32)
(Nproj, Nray, Nslice) = tiltSeries.shape 

# Initialize C++ Object.. 
obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

for s in range(Nslice):
	obj.tiltSeries(tiltSeries[:,:,s].ravel(), s)

# Generate Tilt Angles.
tiltAngles = np.linspace(-75, 75, 76, dtype=np.float32)

# Generate measurement matrix
obj.parallelRay(Nray, tiltAngles)
obj.rowInnerProduct()

recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32)

#Main Loop
for i in range(Niter): 

    print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    for s in range(Nslice):
        recon[:,:,s] = obj.recon(recon[:,:,s].ravel(), beta, s, -1)

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta = beta*beta_red 

# Save the Reconstruction.
np.save('recon.npy', recon)