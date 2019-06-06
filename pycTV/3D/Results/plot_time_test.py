import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from skimage import io
import os.path

# Data Tolerance Parameter
eps = 0.5

# 
Nproj = list(range(1,77))
Niter = np.load('Time/Niter.npy')
tv = np.array([])
dd = np.array([])

for j in range(len(Nproj)):


	i = Nproj[j]
	ind = Niter[j]

	#Read the Data. 
	temp_tv = np.load('Time/' + str(i) + '/tv.npy')
	temp_dd = np.load('Time/' + str(i) + '/dd.npy')

	temp_tv = temp_tv[temp_tv != 0]
	temp_dd = temp_dd[temp_dd != 0]

	tv = np.append(tv, temp_tv)
	dd = np.append(dd, temp_dd)

	if ( i == 76):
		recon = np.load('Time/' + str(i) + '/recon.npy')
		im = recon[134,:,:]/np.amax(recon[134,:,:])
		io.imsave('Time/' + str(i) + '/slice.tif', im)


x = np.arange(tv.shape[0]) + 1

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,4))
fig.subplots_adjust(hspace=0.4)

ax1.plot(x, tv,color='blue', linewidth=2.0)
ax1.set_title('Min TV: ' +str(np.amin(tv)), loc='right', fontsize=10)
ax1.set_title('TV', loc='center', fontweight='bold')
ax1.set_xticklabels([])

ax2.plot(x,dd,color='black', linewidth=2.0)
ax2.axhline(y=eps, color='r')
ax2.set_title('Min dd: ' +str(np.amin(dd)), loc='right', fontsize=10)
ax2.set_title('DD', loc='center', fontweight='bold')
ax2.set_xlabel('Number of Iterations', fontweight='bold')

# plt.savefig('Time/dd_tv_plot.png')

plt.figure(figsize=(8,3))
plt.plot(Nproj, Niter, color='red', linewidth=2.0)
plt.title('Number of Iterations', loc='left', fontweight='bold')
plt.xlabel('Number of Projections', fontweight='bold')
plt.yticks(np.arange(min(Niter), max(Niter)+1, 1))
plt.tight_layout()

# plt.savefig('Time/Niter_plot.png')
# plt.show()