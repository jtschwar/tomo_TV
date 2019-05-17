import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tvlib import tv_derivative2D, parallelRay, tv2D

Niter = 50
num_tilts = 30
beta_0 = 1.0
beta_red = 0.995

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
# b = b.reshape(num_tilts, 256)
# print(np.amax(b))

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

    beta = beta*beta_red


plt.imshow(recon,cmap='gray')
plt.axis('off')
plt.show()