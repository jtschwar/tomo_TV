# Script to simulate tomography reconstructions when a new projection 
# is added every XX Iterations. Instead of Iter_TV that runs each projection 
# for a fixed number of iterations (sequentially), here we run a single projection
# until convergence (~125 iterations) and use the time elapsed to determine
# how many more projections to add/ ('sample').
# Used for simulated datasets (original volume / object is provided)

from tqdm import tqdm
from pytvlib import *
import numpy as np
import plotter
import ctvlib
import logger

##############################################################

fName = 'bowtie'
localDirectory = '/media/hlab/hvol1/Jonathan/20201101_dynamicSIRT'
remoteMonitorDirectory = "/Volumes/Old EMAL Server Data/NewEMALServer2/Talos/Hovden_Group/JSchwartz/AutomatedTomography/20201101_dynamicSIRT"

#Algorithm 
alg = 'SIRT'

# Descent Parameter (for ART / SART)
beta0, beta_red = 0.5, 0.985

# Max Number of iterations before next projection is collected. 
max_iter = 100

# TV Parameter / Reduction.
alpha, alpha_red = 0.2, 0.95

# Data Tolerance Parameter
eps = 0.43

# Number of Iterations (TV Loop)
ng = 10

# Reduction Criteria
r_max = 0.95

#Outcomes:
show_live_plot = True

# Experiment
theta_min, theta_max, dtheta = -76, 76, 1
fileExtension = '.dm4'

###################################################################

# Logger to Read Directory 
tomoLogger = logger.logger(localDirectory, fileExtension)
tomoLogger.beginSFTP(remoteMonitorDirectory)
(Nslice, Nray, Nproj) = tomoLogger.log_projs.shape

# Initialize C++ Object.. 
tomo = ctvlib.ctvlib(Nslice, Nray, Nproj)
tomoLogger.load_tilt_series(tomo)

# Generate measurement matrix
initialize_algorithm(tomo, alg, Nray, tomoLogger.log_tilts)
if alg == 'SIRT': beta = 1/tomo.lipschits()

#Store all the meta data for saving.
fDir = fName + '_dynamicCS'
meta = {'Niter':max_iter,'ng':ng,'beta':beta0,'beta_red':beta_red,'eps':eps}
meta.update({'alpha':alpha,'alpha_red':alpha_red,'r_max':r_max,'alg':alg})

# Check if volume exists? If so, load volume + pass to tomo and load previous results.
fullDD = tomoLogger.load_results(fName, tomo)[0]

if show_live_plot: pr = plotter.plotter() # Initialize Plot

#Dynamic Tilt Series Loop. 
ii = len(tomoLogger.log_tilts)
Nproj_estimate = np.arange(theta_min,theta_max,dtheta).shape[0]
while ii < Nproj_estimate:

    print('Reconstructing Tilt Angles: 0 -> ' + str(ii+1) + ' / ' + str(Nproj))

    if alg != 'SIRT':  beta = beta0 * (1.0 - 2/3 * ii/Nproj)

    dd_vec, tv_vec = np.zeros(max_iter, dtype=np.float32), np.zeros(max_iter, dtype=np.float32)
 
    #Main Reconstruction Loop
    for jj in tqdm(range(max_iter)): 

        tomo.copy_recon()

        run(tomo, alg, beta)
        if alg != 'SIRT': beta *= beta_red

        #Measure Magnitude for TV - GD.
        if (Niter[0] == 0):
            dPOCS0 = tomo.matrix_2norm() * alpha
            dp = dPOCS0 / alpha
        else: # Measure change from ART.
            dp = tomo.matrix_2norm() 

        if (jj == 0): dPOCS = dPOCS0

        # Measure difference between exp/sim projections.
        dd_vec[jj] = tomo.data_distance()

        # Measure TV
        tv_vec[jj] = tomo.tv()

        tomo.copy_recon()

        # TV Minimization. 
        tomo_obj.tv_gd(ng, dPOCS)
        dg = tomo_obj.matrix_2norm()

        if (dg > dp * r_max and dd_vec[jj] > eps):
            dPOCS *= alpha_red

    # Append Results
    fullDD, fullTV = np.append(fullDD, dd_vec), np.apend(fullTV, tv_vec)

    # Run Logger to see how many projections were collected since last check.
    if tomoLogger.check_for_new_tilts():
        # Checkpoint save with all the meta data.
        results = {'fullDD':fullDD, 'fullTV':fullTV}
        tomoLogger.save_results(fName, tomo, meta, results)

        # Update tomo (C++) with new projections / tilt Angles.
        angleStart = tomoLogger.log_tilts[ii]
        initialize_algorithm(tomo, alg, Nray, tomoLogger.log_tilts, angleStart)
        load_exp_tilt_series(tomo, tomoLogger.log_projs)
        # prevTilt = tomoLogger.update_projection_angles(tomo, tomoLogger.log_tilts)
        # initialize_algorithm(tomo, alg, Nray, tomoLogger.log_tilts, prevTilt)

        ii = len(tomoLogger.log_tilts)

print('Experiment Complete!')
tomoLogger.sftp_connection.close()
