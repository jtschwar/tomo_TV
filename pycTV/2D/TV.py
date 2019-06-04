import numpy as np
import ctvlib 
from pytvlib import tv2D, tv_derivative2D
from PIL import Image
from matplotlib import pyplot as plt

########################################

# Number of Iterations (Main Loop)
Niter = 100

# Number of Iterations (TV Loop)
ng = 5

# Step Size for Theta
dTheta = 2

# Parameter in ART Reconstruction.
beta = 1.0

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 0

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

##########################################

#Read Image. 
tiltSeries = Image.open('phantom.tif')
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

b = np.transpose(A.dot(tiltSeries))
recon = np.zeros([Nx, Ny], dtype=np.float32)

#Main Loop
for i in range(Niter): 

    temp_recon = recon.copy()

    if (i % 10 == 0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    obj.recon(np.ravel(recon), b, beta)    

    #Positivity constraint 
    recon[recon < 0] = 0  

    #ART-Beta Reduction
    beta = beta*beta_red 

    g = A.dot(np.ravel(recon))

    if (i == 0):
        dPOCS = np.linalg.norm(recon - temp_recon) * alpha

    dd = np.linalg.norm(g - b)
    dp = np.linalg.norm(recon - temp_recon)   
    temp_recon = recon.copy()

    for j in range(ng):
        v = tv_derivative2D(recon)
        v /= np.linalg.norm(v)
        recon -= dPOCS * v

    dg = np.linalg.norm(recon - temp_recon) 

    if (dg > dp * r_max and dd > eps):
        dPOCS *= alpha_red


# Display the Reconstruction. 
plt.imshow(recon,cmap='gray')
plt.axis('off')
plt.show()