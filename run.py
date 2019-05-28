import shutil
import subprocess

#Copy the Parameters to Results Folder
shutil.copy('parameters.txt', 'Results/parameters.txt')

# Run ASD - POCS Simulation.
print('\nRunning ASD - POCS\n')
subprocess.call('./asd_tv')

# Run Dynamic TV Simulation. 
print('\nRunning Dynamic TV')
subprocess.call('./dynamic_tv')
