# General 3D - ASD/TV Reconstruction with Positivity Constraint. 
# Intended for simulated datasets to measure RMSE and Volume's Original TV. 

import sys, os
sys.path.append('./Utils')
from pytvlib import *
import astra_ctvlib
import numpy as np
import time
import sys

import wandb
wandb.init(project='gpuTomography')
########################################

vol_size = '256_'
file_name = 'au_sto.h5'

# Number of Iterations (Main Loop)
Niter = 10

# Number of Iterations (TV Loop)
ng = 10

# Parameter in ART Reconstruction.
beta0 = 0.25

# ART Reduction.
beta_red = 0.985

# Data Tolerance Parameter
eps = 0.019

# Reduction Criteria
r_max = 0.95
alpha_red = 0.95
alpha = 0.2

SNR = 0

#Outcomes:
show_live_plot = 0
saveGif, saveRecon = False, True           
gif_slice = 156
##########################################

#Read Image. 
(fName, tiltAngles, original_volume) = load_h5_data(vol_size,file_name)
(Nslice, Nray, _) = original_volume.shape
Nproj = tiltAngles.shape[0]

print('Loaded h5 file, now intiializing c++ object')

# Initialize C++ Object.. 
tomo_obj = astra_ctvlib.astra_ctvlib(Nslice, Nray, Nproj, np.deg2rad(tiltAngles))
tomo_obj.initilizeInitialVolume()

# Load Volume and Collect Projections. 
for s in range(Nslice):
    tomo_obj.setOriginalVolume(original_volume[s,:,:],s)

print('Loaded Object, now creating projections')

tomo_obj.create_projections()
tomo_obj.initializeSART('random')

b = np.zeros([Nray,Nray*Nproj],dtype=np.float32)
b[:] = tomo_obj.get_projections()

# Apply poisson noise to volume.
if SNR != 0:
    tomo_obj.poissonNoise(SNR)

#Measure Volume's Original TV, Initalize vectors. 
beta = beta0

print('Measuring TV0')
tv0 = tomo_obj.original_tv()
dd_vec , tv_vec = np.zeros(Niter), np.zeros(Niter) 
rmse_vec, time_vec = np.zeros(Niter), np.zeros(Niter)

# Measure time elapsed. 
counter = 1 
t0 = time.time()

#Main Loop
for i in range(Niter): 


    if ( i % 25 ==0):
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))

    tomo_obj.copy_recon()

    #ART Reconstruction. 
    tomo_obj.SART(beta)

    #ART-Beta Reduction
    beta *= beta_red 

    #Forward Projection.
    tomo_obj.forwardProjection()

    #Measure Magnitude for TV - GD.
    if (i == 0):
        dPOCS = tomo_obj.matrix_2norm() * alpha
        dp = dPOCS / alpha
    else: # Measure change from ART.
        dp = tomo_obj.matrix_2norm() 

    # Measure difference between exp/sim projections.
    dd_vec[i] = tomo_obj.vector_2norm()

    #Measure TV. 
    tv_vec[i] = tomo_obj.tv()

    #Measure RMSE.
    rmse_vec[i] = tomo_obj.rmse()

    tomo_obj.copy_recon() 

    #TV Minimization. 
    tomo_obj.tv_gd(ng, dPOCS)
    dg = tomo_obj.matrix_2norm()

    if(dg > dp * r_max and dd_vec[i] > eps):
        dPOCS *= alpha_red

    if (i+1)% 25 == 0:
        timer(t0, counter, Niter)
        if show_live_plot:
            pr.sim_ASD_live_plot(dd_vec, eps, tv_vec, tv0, rmse_vec, i)
    counter += 1
    time_vec[i] = time.time() - t0


print('RMSE: ' + str(rmse_vec))
print('TV: ' + str(tv_vec))
print('DD: ' + str(dd_vec))

print('Reconstruction Complete, Saving Data..')
print('Save Gif :: {}, Save Recon :: {}'.format(saveGif, saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_ASD'
meta = {'Niter':Niter,'ng':ng,'beta':beta0,'beta_red':beta_red,'eps':eps,'r_max':r_max}
meta.update({'alpha':alpha,'alpha_red':alpha_red,'SNR':SNR,'vol_size':vol_size})
results = {'dd':dd_vec,'eps':eps,'tv':tv_vec,'tv0':tv0,'rmse':rmse_vec,'time':time_vec}
save_results([fDir,fName], meta, results)

if saveGif: 
    save_gif([fDir,fName], gif_slice, gif)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)
