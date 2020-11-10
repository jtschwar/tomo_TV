# Script to simulate tomography reconstructions when a new projection
# is added every XX Iterations. Instead of Iter_TV that runs each projection
# for a fixed number of iterations (sequentially), here we run a single projection
# until convergence (~125 iterations) and use the time elapsed to determine
# how many more projections to add/ ('sample').
# Used for simulated datasets (original volume / object is provided)

import mpi_astra_ctvlib
from tqdm import tqdm
from pytvlib import *
import numpy as np
import plotter
import mpi_logger
check_cuda()
########################################

fName = 'bowtie'
localDirectory = '/media/hlab/hvol1/Jonathan/20201018_dynamicCS'
remoteMonitorDirectory = "/Volumes/Old EMAL Server Data/NewEMALServer2/Talos/Hovden_Group/JSchwartz/AutomatedTomography/20201018_dynamicCS_v2"

# Algorithm
alg = 'SART'
initAlg = 'random'

# Max Number of iterations before next projections are appended.
max_iter = 50

# ART Parameter / Reduction.
beta0, beta_red = 0.5, 0.99

# TV Parameter / Reduction.
alpha, alpha_red = 0.2, 0.95

# Data Tolerance Parameter
eps = 0.43

# Number of Iterations (TV Loop)
ng = 10

# Reduction Criteria
r_max = 0.95

#Outcomes:
show_live_plot, saveRecon = True, True

# Experiment
theta_min, theta_max, dtheta = -76, 76, 1
fileExtension = '.dm4'

###################################################################

# Logger to Read Directory
tomoLogger = mpi_logger.logger(localDirectory,fileExtension)
tomoLogger.beginSFTP(remoteMonitorDirectory)
tiltAngles = tomoLogger.log_tilts

(Nslice, Nray, Nproj) = tomoLogger.log_projs.shape

# Initialize C++ Object..
tomo = mpi_astra_ctvlib.mpi_astra_ctvlib(Nslice, Nray, Nproj, np.deg2rad(tiltAngles))
initialize_algorithm(tomo, alg, initAlg)
tomoLogger.load_tilt_series_mpi(tomo)
tomo.initializeReconCopy()

#Store all the meta data for saving.
fDir = fName + '_dynamicCS'
meta = {'Niter':max_iter,'ng':ng,'beta':beta0,'beta_red':beta_red,'eps':eps}
meta.update({'alpha':alpha,'alpha_red':alpha_red,'r_max':r_max,'alg':alg,'initAlg':initAlg})

# Check if volume exists? If so, load volume + pass to tomo and load previous results.
(fullDD, fullTV) = tomoLogger.load_results(fName, tomo)

if show_live_plot and tomo.rank() == 0: pr = plotter.plotter()

#Dynamic Tilt Series Loop.
ii = len(tomoLogger.log_tilts)
Nproj_estimate = np.arange(theta_min,theta_max,dtheta).shape[0]
while ii < Nproj_estimate:

    if tomo.rank() == 0: print('\nReconstructing Tilt Angles: 0 -> ' + str(ii+1) + ' / ' + str(Nproj_estimate))

    beta = beta0 * (1.0 - 2/3 * ii/Nproj_estimate)     # Reset Beta.

    dd_vec, tv_vec = np.zeros(max_iter, dtype=np.float32), np.zeros(max_iter, dtype=np.float32)

    #Main Reconstruction Loop
    for jj in tqdm(range(max_iter)):

        tomo.copy_recon()

        run(tomo, alg, beta)     # Reconstruction

        beta *= beta_red         # Beta Reduction.

        #Measure Magnitude for TV - GD.
        if (ii == 0 and jj == 0):
            dPOCS0 = tomo.matrix_2norm() * alpha
            dp = dPOCS0 / alpha
        else: # Measure change from ART.
            dp = tomo.matrix_2norm()
            
        if (jj == 0): dPOCS = dPOCS0

        # Measure difference between exp/sim projections.
        tomo.forwardProjection()
        dd_vec[jj] = tomo.vector_2norm()

        #TV Minimization.
        tomo.copy_recon()
        tv_vec[jj] = tomo.tv_gd(ng, dPOCS)
        dg = tomo.matrix_2norm()

        if (dg > dp * r_max and dd_vec[jj] > eps):
            dPOCS *= alpha_red

    # Append Results
    fullDD, fullTV = np.append(fullDD, dd_vec), np.append(fullTV, tv_vec)

    if show_live_plot and tomo.rank() == 0: # Show live plot.
        pr.dynamicCS_live_plot(tomo, tomoLogger, fullDD, eps, fullTV)

    # Run Logger to see how many projections were collected since last check.
    if tomoLogger.check_for_new_tilts():

        # Checkpoint save with all the meta data.
        results = {'fullDD':fullDD, 'fullTV':fullTV}
        tomoLogger.save_results_mpi(fName, tomo, meta, results)

        # Update tomo (C++) with new projections / tilt Angles.
        tomo.update_projection_angles(np.deg2rad(tomoLogger.log_tilts))
        tomoLogger.load_tilt_series_mpi(tomo)
        initialize_algorithm(tomo, alg, initAlg)

        ii = len(tomoLogger.log_tilts)

# Finalize and Close SFTP Connection
if tomo.rank() == 0: 
    print('Experiment Complete!')
    tomoLogger.sftp_connection.close()
tomo.finalize()
