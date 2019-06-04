import sys
sys.path.append('./Utils')
from pytvlib import tv, tv_derivative
import ctvlib 
from skimage import io
import numpy as np
########################################

# Number of Iterations (TV Loop)
ng = 20

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

#Transfer Tilt Series to C++ Object. 
for s in range(Nslice):
    b[:,s] = tiltSeries[:,:,s].ravel()
obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.linspace(-75, 75, 76, dtype=np.float32)

# Generate measurement matrix
obj.parallelRay(Nray, tiltAngles)
obj.rowInnerProduct()

#Generate Reconstruction. 
recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32)
tv_vec = np.zeros()

#Dynamic Tilt Series Loop. 
for i in range(180):

    print('Reconstructing Tilt Angles: 0 -> ' + str(i+1) )

    tv_vec = np.zeros(100)
    dd_vec = np.zeros(100)

    k = 0
 
    #Main Reconstruction Loop
    while True: 

        temp_recon = recon.copy()

        #ART Reconstruction. 
        for s in range(Nslice):
            recon[:,:,s] = obj.recon(recon[:,:,s].ravel(), beta, s, Nray) 

        #Positivity constraint 
        recon[recon < 0] = 0  

        #ART-Beta Reduction.
        beta = beta*beta_red 

        #Forward Projection/
        for s in range(Nslice):
            g[:,s] = obj.forwardProjection(recon[:,:,s].ravel())

        if (i == 0):
            dPOCS = np.linalg.norm(recon - temp_recon) * alpha

        dd_vec[k] = np.linalg.norm(g - b)
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

        if time > time_limit:
            break


    # Save Data. 
    os.mkdir('Results/Time/' + str(i))
    np.save('Results/Time/' + str(i) + '/tv.npy', tv_vec)
    np.save('Results/Time/' + str(i) + '/dd.npy', dd_vec)
    np.save('Results/Time/' + str(i) + '/recon.npy', recon)

# Save the Reconstruction.
np.save('Results/Co2P_recon.npy', recon)