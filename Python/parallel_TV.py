#Implementation that uses multiple processes to parallelize the ART loop. 

import numpy as np
from tvlib import tv_derivative, tv, parallelRay, generate_tilt_series, tomography
import multiprocessing as mp
import time

Niter = 25
ng = 10
beta = 1.0
r_max = 1.0
gamma_red = 0.8
beta_red = 0.95
ncores = int(mp.cpu_count()/2)

tiltSeries = np.load('Co2P_projections.npy')
tiltAngles = np.load('tiltAngles.npy')
(Nslice, Nray, Nproj) = tiltSeries.shape

# Generate measurement matrix
A = parallelRay(Nray, 1.0, tiltAngles, Nray, 1.0) #A is a sparse matrix
A = A.tocsr()
(Nrow, Ncol) = A.shape

recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F')
row = np.zeros(Ncol, dtype=np.float32)
rowInnerProduct = np.zeros(Nrow, dtype=np.float32)
residual = np.zeros(Niter)

for j in range(Nrow):
    row[:] = A[j,:].toarray()
    rowInnerProduct[j] = np.dot(row,row)
row = None

t0 = time.time()
counter = 1

#Main Loop
for i in range(Niter): 

    print('Iteration No.: ' + str(i+1) +'/'+str(Niter))
    
    recon_temp = recon.copy()
    recon_residual = recon.copy()

    #ART   
    with mp.Pool(processes=ncores) as pool:
        recon[:] = pool.starmap(tomography, [(recon[s,:,:],tiltSeries[s,:,:],
            A,beta, rowInnerProduct) for s in range(Nslice)])

    #Positivity constraint 
    recon[recon < 0] = 0  

    beta = beta*beta_red #ART-Beta Reduction

    #calculate tomogram change due to POCS
    if i == 0:
        dPOCS = np.linalg.norm(recon_temp - recon)

    dp = np.linalg.norm(recon_temp - recon)

    recon_temp = recon.copy() 

    #3D TV minimization
    for j in range(ng):
        R_0 = tv(recon)
        v = tv_derivative(recon)
        recon_prime = recon - dPOCS * v
        recon_prime[recon_prime < 0] = 0
        gamma = 1.0
        R_f = tv(recon_prime)

        #Projected Line search
        while R_f > R_0:
            gamma = gamma * gamma_red
            recon_prime = recon - gamma * dPOCS * v
            recon_prime[recon_prime < 0] = 0
            R_f = tv(recon_prime)
        recon[:] = recon_prime

    dg = np.linalg.norm(recon - recon_temp)
    residual[i] = np.linalg(recon - recon_residual)

    if dg > r_max*dp:
        dPOCS = dPOCS*0.95

    timeLeft = (time.time() - t0)/counter * (Niter - counter)
    counter += 1
    timeLeftMin, timeLeftSec = divmod(timeLeft, 60)
    timeLeftHour, timeLeftMin = divmod(timeLeftMin, 60)
    print('Estimated time to complete: %02d:%02d:%02d' % (timeLeftHour, timeLeftMin, timeLeftSec))

   if i%5==0:
        np.save('recon_'+str(i)+'.npy',recon)

np.save('final_recon.npy', recon)
np.save('residual.npy', residual)