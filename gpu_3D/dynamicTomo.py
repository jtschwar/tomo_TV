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
localDirectory = '/media/hlab/hvol1/Jonathan/20201101_dynamicSIRT'
remoteMonitorDirectory = "/Volumes/Old EMAL Server Data/NewEMALServer2/Talos/Hovden_Group/JSchwartz/AutomatedTomography/20201101_dynamicSIRT"

# Algorithm
alg = 'SIRT'
initAlg = ''

# Max Number of iterations before next projections are appended.
max_iter = 100

#Outcomes:
show_live_plot, saveRecon = True, True

# Experiment
theta_min, theta_max, dtheta = -76, 76, 1

###################################################################

# Logger to Read Directory
tomoLogger = mpi_logger.logger(localDirectory)
tomoLogger.beginSFTP(remoteMonitorDirectory)
tiltAngles = tomoLogger.log_tilts

(Nslice, Nray, Nproj) = tomoLogger.log_projs.shape

# Initialize C++ Object..
tomo = mpi_astra_ctvlib.mpi_astra_ctvlib(Nslice, Nray, Nproj, np.deg2rad(tiltAngles))
initialize_algorithm(tomo, alg, initAlg)
tomoLogger.load_tilt_series_mpi(tomo)

#Store all the meta data for saving.
meta = {'alg':alg,'initAlg':initAlg,'Niter':max_iter}

# Check if volume exists? If so, load volume + pass to tomo and load previous results.
fullDD = tomoLogger.load_results(fName, tomo)[0]

if show_live_plot and tomo.rank() == 0: pr = plotter.plotter()

#Dynamic Tilt Series Loop.
ii = len(tomoLogger.log_tilts)
Nproj_estimate = np.arange(theta_min,theta_max,dtheta).shape[0]
while ii < Nproj_estimate:

    if tomo.rank() == 0: print('\nReconstructing Tilt Angles: 0 -> ' + str(ii+1) + ' / ' + str(Nproj_estimate))

    dd_vec = np.zeros(max_iter, dtype=np.float32)

    #Main Reconstruction Loop
    for jj in tqdm(range(max_iter)):

        run(tomo, alg)     # Reconstruction

        # Measure difference between exp/sim projections.
        tomo.forwardProjection()
        dd_vec[jj] = tomo.vector_2norm()

    # Append Results
    fullDD = np.append(fullDD, dd_vec)

    # Run Logger to see how many projections were collected since last check.
    newData = tomoLogger.check_for_new_tilts()
    if newData:
        # Checkpoint save with all the meta data.
        results = {'fullDD':fullDD}
        tomoLogger.save_results_mpi(fName, tomo, meta, results)

        # Update tomo (C++) with new projections / tilt Angles.
        tomo.update_projection_angles(tomoLogger.log_tilts)
        tomoLogger.load_tilt_series_mpi(tomo)
        initialize_algorithm(tomo, alg, initAlg)

        ii = len(tomoLogger.log_tilts)

    if show_live_plot and tomo.rank() == 0: # Show live plot.
         pr.dynamicSIRT_live_plot(tomo, tomoLogger, fullDD)

# Finalize and Close SFTP Connection
if tomo.rank() == 0: 
    print('Experiment Complete!')
    tomoLogger.sftp_connection.close()
tomo.finalize()
