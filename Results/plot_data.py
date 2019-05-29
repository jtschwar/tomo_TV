import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
from skimage import io
import os.path

#Theta Max Values. 
index = [20, 40, 60, 80, 100, 120, 140, 160, 180]

# Set Epsilon value. 
eps = 1.0

# Set the original TV value. 
# tv0 = 372273 #phantom
tv0 = 251920 #Co2P - Volume

#Plot Dynamic Figure. 
###################################################

temp = np.loadtxt('20/rmse.txt')
ind = temp.size

rmse = np.zeros(temp.size*9)
tv = np.zeros(temp.size*9)
dd = np.zeros(temp.size*9)
ver_lines = np.zeros(10)
stop = 0
for j in range(len(index)):

	i = index[j]

	temp = np.loadtxt(str(i) + '/rmse.txt')
	ind = temp.size

	start = stop
	stop = ind + start
	
	ver_lines[j+1] = ind + ver_lines[j]

	#Read the Data. 
	rmse[start:stop] = np.loadtxt(str(i)+'/rmse.txt')
	tv[start:stop] = np.loadtxt(str(i)+'/tv.txt')
	dd[start:stop] = np.loadtxt(str(i)+'/dd.txt')

#Remove Zeros
rmse = rmse[rmse != 0]
tv = tv[tv != 0]
dd = dd[dd != 0]

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
for xc in ver_lines:
	ax3.axvline(x=xc, color='orange', linestyle='--')
ax3.axvspan(ver_lines[-2], ver_lines[-1], alpha=0.2, color='green')


fig.suptitle('Dynamic Compressed Sensing', fontsize=14)
fig.subplots_adjust(top=0.9)

# Plot ASD Figure. 
#######################################################

#Read the Data. 
rmse = np.loadtxt('ASD_tv/rmse.txt')
cosAlph = np.loadtxt('ASD_tv/Cos_Alpha.txt')
tv = np.loadtxt('ASD_tv/tv.txt')
dd = np.loadtxt('ASD_tv/dd.txt') 

x = np.arange(tv.shape[0]) + 1

fig, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(6, 6))
fig.subplots_adjust(hspace=0.5)

ax1.plot(x,rmse,color='m',linewidth=2.0)
ax1.set_title('Min RMSE: ' +str(np.amin(rmse)), loc='right', fontsize=10)
ax1.set_title('RMSE', loc='left', fontweight='bold')
ax1.axvspan(0, x[-1], alpha=0.2, color='green')

ax2.plot(x,tv,color='blue',linewidth=2.0)
ax2.axhline(y=tv0, color='r')
ax2.set_title('Final TV: ' +str(tv[-1]), loc='right', fontsize=10)
ax2.set_title('TV', loc='left', fontweight='bold')
ax2.axvspan(0, x[-1], alpha=0.2, color='green')

ax3.plot(x,dd,color='black', linewidth=2.0)
ax3.axhline(y=eps, color='r')
ax3.set_title('Min dd: ' +str(np.amin(dd)), loc='right', fontsize=10)
ax3.set_title('DD', loc='left', fontweight='bold')
ax3.set_xlabel('Number of Iterations', fontweight='bold')
ax3.axvspan(0, x[-1], alpha=0.2, color='green')

fig.suptitle('Fixed Batch (ASD - POCS)', fontsize=14)
fig.subplots_adjust(top=0.9) 


# Plot last Epoch for Dynamic Data (for direct comparison).
############################################################

rmse = np.loadtxt('180/rmse.txt')
tv = np.loadtxt('180/tv.txt')
dd = np.loadtxt('180/dd.txt')

x = np.arange(tv.shape[0]) + 1

fig, (ax1,ax2, ax3) = plt.subplots(3,1, figsize=(6, 6))
fig.subplots_adjust(hspace=0.3)

ax1.plot(x,rmse,color='m',linewidth=2.0)
ax1.set_title('Min RMSE: ' +str(np.amin(rmse)), loc='right', fontsize=10)
ax1.set_title('RMSE', loc='left', fontweight='bold')
ax1.set_xticklabels([])

ax2.plot(x,tv,color='blue',linewidth=2.0)
ax2.axhline(y=tv0, color='r')
ax2.set_title('Final TV: ' +str(tv[-1]), loc='right', fontsize=10)
ax2.set_title('TV', loc='left', fontweight='bold')
ax2.set_xticklabels([])

ax3.plot(x,dd,color='black', linewidth=2.0)
ax3.axhline(y=eps, color='r')
ax3.set_title('Min dd: ' +str(np.amin(dd)), loc='right', fontsize=10)
ax3.set_title('DD', loc='left', fontweight='bold')
ax3.set_xlabel('Number of Iterations', fontweight='bold')

fig.suptitle('Dynamic Compressed Sensing', fontsize=14)
fig.subplots_adjust(top=0.9) 

# Show Final Reconstructions.
###########################################################################
orig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Test_Images/Co2P_v2.tif')
original_img = plt.imread(orig_dir)
dynamic_recon = io.imread('180/recon.tif')
ASD_recon = io.imread('ASD_tv/recon.tif')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6,4))

ax1.imshow(original_img, cmap='gray')
ax1.set_title('Original Image')
ax1.axis('off')

ax2.imshow(dynamic_recon, cmap='gray')
ax2.set_title('Dynamic CS')
ax2.axis('off')

ax3.imshow(ASD_recon, cmap='gray')
ax3.set_title('Traditional ASD/POCS')
ax3.axis('off')

plt.show() 