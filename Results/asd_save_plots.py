import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

# Set Epsilon value. 
eps = 0

# Set Original TV value. 
tv0 = 372273

#Read the Data. 
rmse = np.loadtxt('ASD_tv/rmse.txt')
cosAlph = np.loadtxt('ASD_tv/Cos_Alpha.txt')
tv = np.loadtxt('ASD_tv/tv.txt')
dd = np.loadtxt('ASD_tv/dd.txt')

x = np.arange(tv.shape[0]) + 1

fig, (ax1,ax2, ax3, ax4) = plt.subplots(4,1, figsize=(6, 6))
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

plt.savefig('ASD_tv/plot.png')