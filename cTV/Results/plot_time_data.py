import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
from skimage import io
import os.path

#Theta Max Values. 
Nproj = list(range(1,181))

# Set Epsilon value. 
eps = 1.0

# Set the original TV value. 
# tv0 = 372273 #phantom
tv0 = 251920 #Co2P - Volume

#Plot Dynamic Figure. 
###################################################

Niter = np.loadtxt('180/Niter.txt')

rmse = np.array([])
tv = np.array([])
dd = np.array([])
ver_lines = np.zeros(len(Niter)+1)

stop = 0
for j in range(len(Nproj)):

	i = Nproj[j]
	ind = Niter[j]
	
	ver_lines[j+1] = ind + ver_lines[j]

	temp_rmse = np.loadtxt(str(i)+'/rmse.txt')
	temp_tv = np.loadtxt(str(i)+'/tv.txt')
	temp_dd = np.loadtxt(str(i)+'/dd.txt')

	temp_rmse = temp_rmse[temp_rmse != 0]
	temp_tv = temp_tv[temp_tv != 0]
	temp_dd = temp_dd[temp_dd != 0]

	#Read the Data. 
	rmse = np.append(rmse, temp_rmse)
	tv = np.append(tv, temp_tv)
	dd = np.append(dd, temp_dd)

x = np.arange(rmse.shape[0]) + 1

fig, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(7, 6))
fig.subplots_adjust(hspace=0.5)

ax1.plot(x,rmse,color='m',linewidth=2.0)
ax1.set_title('Min RMSE: ' +str(np.amin(rmse)), loc='right', fontsize=10)
ax1.set_title('RMSE', loc='left', fontweight='bold')
ax1.axvspan(ver_lines[-2], ver_lines[-1], alpha=0.2, color='green')

ax2.plot(x,tv,color='blue',linewidth=2.0)
ax2.axhline(y=tv0, color='r')
ax2.set_title('Final TV: ' +str(tv[-1]), loc='right', fontsize=10)
ax2.set_title('TV', loc='left', fontweight='bold')
ax2.axvspan(ver_lines[-2], ver_lines[-1], alpha=0.2, color='green')

ax3.plot(x,dd,color='black', linewidth=2.0)
ax3.axhline(y=eps, color='r')
ax3.set_title('Min dd: ' +str(np.amin(dd)), loc='right', fontsize=10)
ax3.set_title('DD', loc='left', fontweight='bold')
ax3.set_xlabel('Number of Iterations', fontweight='bold')
for xc in ver_lines[0:13]:
	ax3.axvline(x=xc, color='orange', linestyle='--', linewidth=1)
ax3.axvspan(ver_lines[-2], ver_lines[-1], alpha=0.2, color='green')


fig.suptitle('Dynamic Compressed Sensing', fontsize=14)
fig.subplots_adjust(top=0.9)

plt.show()