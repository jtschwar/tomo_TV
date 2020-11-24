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

#Outcomes:
show_live_plot = True

# Experiment
theta_min, theta_max, dtheta = -76, 76, 1
fileExtension = '.dm4'

###################################################################

# Logger to Read Directory 
tomoLogger = logger.logger(localDirectory, fileExtension)
tomoLogger.beginSFTP(remoteMonitorDirectory)
tiltAngles = tomoLogger.log_tilts

(Nslice, Nray, Nproj) = tomoLogger.log_projs.shape

# Initialize C++ Object.. 
tomo = ctvlib.ctvlib(Nslice, Nray, Nproj)
tomoLogger.load_tilt_series(tomo)

# Generate measurement matrix
initialize_algorithm(tomo, alg, Nray, tomoLogger.log_tilts)
if alg == 'SIRT': beta = 1/tomo.lipschits()

#Store all the meta data for saving.
meta = {'alg': alg, 'Niter':max_iter}

# Check if volume exists? If so, load volume + pass to tomo and load previous results.
fullDD = tomoLogger.load_results(fName, tomo)[0]

if show_live_plot: pr = plotter.plotter() # Initialize Plot

#Dynamic Tilt Series Loop. 
ii = len(tomoLogger.log_tilts)
Nproj_estimate = np.arange(theta_min,theta_max,dtheta).shape[0]
while ii < Nproj_estimate:

    print('Reconstructing Tilt Angles: 0 -> ' + str(i+1) + ' / ' + str(Nproj))

    if alg != 'SIRT': beta = beta0
    dd_vec = np.zeros(max_iter, dtype=np.float32)
 
    #Main Reconstruction Loop
    for jj in tqdm(range(max_iter)): 

        run(tomo, alg, beta)
        if alg != 'SIRT': beta *= beta_red

        # Measure difference between exp/sim projections.
        dd_vec[jj] = tomo.data_distance()

    # Append Results
    fullDD = np.append(fullDD, dd_vec)

    # Run Logger to see how many projections were collected since last check.
    if tomoLogger.check_for_new_tilts():
        # Checkpoint save with all the meta data.
        results = {'fullDD':fullDD}
        tomoLogger.save_results(fName, tomo, meta, results)

        # Update tomo (C++) with new projections / tilt Angles.
        tomo.update_projection_angles(tomoLogger.log_tilts)
        tomoLogger.load_tilt_series(tomo)
        initialize_algorithm(tomo, alg, initAlg)

        ii = len(tomoLogger.log_tilts)

print('Experiment Complete!')
tomoLogger.sftp_connection.close()
