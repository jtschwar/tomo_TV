import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tvlib import tv_derivative2D, parallelRay, tv2D

Niter = 20
num_tilts = 30
ng = 5
beta_0 = 1.0
r_max = 1.0
gamma_red = 0.8
beta_red = 0.9

beta = beta_0

tiltSeries = Image.open('phantom.tif')
tiltSeries = np.array(tiltSeries)

(Nx, Ny) = tiltSeries.shape
tiltSeries = tiltSeries.flatten()

# Generate Tilt Angles.
tiltAngles = np.linspace(0, 180, num_tilts)

# Generate measurement matrix
A = parallelRay(Nx, 1.0, tiltAngles, Ny, 1.0) #A is a sparse matrix
recon = np.zeros([Nx, Ny], dtype=np.float32, order='F')
A = A.todense()
b = np.transpose(np.dot(A,tiltSeries))

(Nrow, Ncol) = A.shape

rowInnerProduct = np.zeros(Nrow, dtype=np.float32)
row = np.zeros(Ncol, dtype=np.float32)
f = np.zeros(Ncol, dtype=np.float32) # Placeholder for 2d image

# Calculate row inner product, preparation for ART recon
for j in range(Nrow):
    row[:] = A[j, :]
    rowInnerProduct[j] = np.dot(row, row)

for i in range(Niter): #main loop

    recon_temp = recon.copy()

    f[:] = recon.flatten()

    print('Iteration No. '+str(i+1)+'/'+str(Niter))

    for j in range(Nrow):
        row[:] = A[j, :]
        a = (b[j] - np.dot(row, f)) / rowInnerProduct[j]
        f = f + row * a[0,0] * beta
    recon = f.reshape(Nx, Nx)

    recon[recon < 0] = 0 #Positivity constraint  

    print(np.amax(recon))

    beta = beta*beta_red

    #calculate tomogram change due to POCS
    if i == 0:
        dPOCS = np.linalg.norm(recon_temp - recon)

    dp = np.linalg.norm(recon_temp - recon)

    recon_temp = recon.copy() 

    for j in range(5):
        R_0 = tv2D(recon)
        v = tv_derivative2D(recon)
        recon_prime = recon - dPOCS * v
        recon_prime[recon_prime < 0] = 0
        gamma = 1.0
        R_f = tv2D(recon_prime)

        #Projected Line search
        # while R_f > R_0:
        #     gamma = gamma * gamma_red
        #     recon_prime = recon - gamma * dPOCS * v
        #     recon_prime[recon_prime < 0] = 0
        #     R_f = tv2D(recon_prime)
        recon[:] = recon_prime

plt.imshow(recon,cmap='gray')
plt.axis('off')
plt.show()