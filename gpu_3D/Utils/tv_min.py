import regularization
import numpy as np
import h5py
import sys

alg = 'tv_gd'


# Load Data (ie with h5py)

vol = load...

# Set Parameters (if fgp ~ epsilon 0.2, if gd calc with euc - norm )
epsilon = 0.2
ng = 20

print('Loading Reconstruction')
recon = np.array(io.imread(dir+fname+'.tiff'), dtype=np.float32)
(nx,ny,nz) = recon.shape

print('Passing Data to C++')
reg_obj = regularization.regularization(nx, ny)
for s in range(nx):
	reg_obj.setRecon(recon[s,:,:],s)
	
print('Minimizing Volumes TV')
if alg == 'tv_gd':
	dPOCS = tv_obj.matrix_2norm() * epsilon
	tv_obj.tv_gd(ng,dPOCS)
elif alg == 'tv_fgp':
	tv_obj.tv_fgp(ng, epsilon)
else:
	print('Incorrect Algorithm Selected.')

print('Saving Data..')
for s in range(nx):
        recon[s,:,:] = tv_obj.getRecon(s)

# Save Data
# sio.savemat('../recon.mat', {'recon':recon})
