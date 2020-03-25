# General 3D - ART Reconstruction with Positivity Constraint. 

import sys
sys.path.append('./Utils')
import plot_results as pr
from pytvlib import *
import numpy as np
import ctvlib 
import time
########################################

vol_size = '256_'
file_name = 'Co2P_tiltser.tif'

# Number of Iterations (Main Loop)
Niter = 100

# Parameter in ART Reconstruction.
beta = 1.0

# ART Reduction.
beta_red = 0.995

saveRecon = False #Save Final Reconstruction
show_live_plot = False #Calculate dd and show intermediate plots.

##########################################

# #Read Image. 
(file_name, tiltSeries) = load_data(vol_size,file_name)
(Nslice, Nray, Nproj) = tiltSeries.shape
b = np.zeros([Nslice, Nray*Nproj])

# Initialize C++ Object.. 
tomo_obj = ctvlib.ctvlib(Nslice, Nray, Nproj)

for s in range(Nslice):
    b[s,:] = tiltSeries[s,:,:].transpose().ravel()
tomo_obj.setTiltSeries(b)
tiltSeries = None

# Generate Tilt Angles.
tiltAngles = np.load('Tilt_Series/'+file_name+'_tiltAngles.npy')

# Generate measurement matrix
A = parallelRay(Nray, tiltAngles)
tomo_obj.load_A(A)
A = None
tomo_obj.rowInnerProduct()

rmse_vec,dd_vec = np.zeros(Niter), np.zeros(Niter)

t0 = time.time()
counter = 1

#Main Loop
for i in range(Niter): 

    tomo_obj.ART(beta, -1)

    #Positivity constraint 
    tomo_obj.positivity()

    #ART-Beta Reduction
    beta *= beta_red 

    if show_final_plot or show_live_plot:
        tomo_obj.forwardProjection(-1)
        dd_vec[i] = tomo_obj.vector_2norm()
        rmse_vec[i] = tomo_ojb.rmse()

    if (i+1) % 25 ==0:
        print('Iteration No.: ' + str(i+1) +'/'+str(Niter))
        timer(t0, counter, Niter)
        if show_live_plot:
            pr.live_ART_results(dd_vec,i)

    counter += 1

print('Reconstruction Complete, Saving Data..')
print('Save Gif :: {}, Save Recon :: {}'.format(saveGif, saveRecon))

#Save all the results to h5 file. 
fDir = fName + '_ART'
create_save_directory([fDir])
meta = {'Niter':Niter,'ng':ng,'beta':beta0,'beta_red':beta_red,'eps':eps,'r_max':r_max}
meta = meta.update({'alpha':alpha,'alpha_red':alpha_red,'SNR':SNR,'vol_size':vol_size})
results = {'dd':dd_vec,'eps':eps,'tv':tv_vec,'tv0':tv0,'rmse':rmse_vec,'time':time_vec}
save_results([fDir,fName], meta, results)

if saveRecon: 
    save_recon([fDir,fName], (Nslice, Nray, Nproj), tomo_obj)
