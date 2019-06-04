import sys
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative
import ctvlib 
from skimage import io
import numpy as np
########################################

# Number of Iterations (Main Loop)
Niter = 20

# Number of Iterations (TV Loop)
ng = 20

# ART Reduction.
beta_red = 0.995

# Data Tolerance Parameter
eps = 0.5

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.3

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

for k in range(10):

    print('Reconstructing with Epsilon = ' + str(eps))

    # Reset Beta.
    beta = 1.0

    #Create Vectors.
    recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32)
    tv_vec = np.zeros(Niter, dtype=np.float32)
    dd_vec = np.zeros(Niter, dtype=np.float32)

    #Main Loop
    for i in range(Niter): 

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

        dd_vec[i] = np.linalg.norm(g - b)
        dp = np.linalg.norm(recon - temp_recon)   
        temp_recon = recon.copy()

        for j in range(ng):
            v = tv_derivative(recon)
            v /= np.linalg.norm(v)
            recon -= dPOCS * v
        tv_vec[i] = tv(recon)

        dg = np.linalg.norm(recon - temp_recon) 

        if (dg > dp * r_max and dd_vec[i] > eps):
            dPOCS *= alpha_red

    eps += 0.1
    eps = round(eps, 1)

    # Save the Reconstruction.
    os.mkdir('Results/Epsilon_Test/' + str(eps))
    np.save('Results/Epsilon_Test/' + str(eps) + '/recon.npy', recon)
    np.save('Results/Epsilon_Test/' + str(eps) + '/tv.npy', tv_vec)
    np.save('Results/Epsilon_Test/' + str(eps) + '/dd.npy', dd_vec)

    