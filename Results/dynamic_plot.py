import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

#Theta Max Values. 
index = [20, 40, 60, 80, 100, 120, 140, 160, 180]

# Set Epsilon value. 
eps = 0

# Set the original TV value. 
# tv0 = 372273 #phantom
tv0 = 251920 #Co2P - Volume

temp = np.loadtxt('20/rmse.txt')
ind = temp.size

rmse = np.zeros(temp.size*9)
tv = np.zeros(temp.size*9)
dd = np.zeros(temp.size*9)
cosAlph = np.zeros(temp.size*9)
ver_lines = np.zeros(10)

for j in range(len(index)):

	i = index[j]

	start = j * ind
	stop = (j+1) * ind

	ver_lines[j+1] = ind*(j+1)

	#Read the Data. 
	rmse[start:stop] = np.loadtxt(str(i)+'/rmse.txt')
	tv[start:stop] = np.loadtxt(str(i)+'/tv.txt')
	dd[start:stop] = np.loadtxt(str(i)+'/dd.txt')
	cosAlph[start:stop] = np.loadtxt(str(i)+'/Cos_Alpha.txt')

x = np.arange(rmse.shape[0]) + 1

fig, (ax1,ax2, ax3, ax4) = plt.subplots(4,1, figsize=(7, 6))
fig.subplots_adjust(hspace=1.0)

ax1.plot(x,rmse,color='m',linewidth=2.0)
ax1.set_title('Min RMSE: ' +str(np.amin(rmse)), loc='right', fontsize=10)
ax1.set_title('RMSE', loc='left', fontweight='bold')

ax2.plot(x,tv,color='green',linewidth=2.0)
ax2.axhline(y=tv0, color='r')
ax2.set_title('Final TV: ' +str(tv[-1]), loc='right', fontsize=10)
ax2.set_title('TV', loc='left', fontweight='bold')

ax3.plot(x,dd,color='black', linewidth=2.0)
ax3.axhline(y=eps, color='r')
ax3.set_title('Min dd: ' +str(np.amin(dd)), loc='right', fontsize=10)
ax3.set_title('DD', loc='left', fontweight='bold')

ax4.plot(x, cosAlph,color='blue',linewidth=2.0)
ax4.set_xlabel('Number of Iterations', fontweight='bold')
ax4.set_title('Min Cosine-Alpha: ' +str(np.amin(cosAlph)), loc='right', fontsize=10)
ax4.set_title('Cosine-Alpha', loc='left', fontweight='bold')
for xc in ver_lines:
	ax4.axvline(x=xc, color='orange', linestyle='--')

plt.show()  