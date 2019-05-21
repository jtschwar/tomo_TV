import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

index = [20, 40, 60, 80, 100, 120, 140, 160, 180]

# Set Directory
i = 180

# Set Epsilon value. 
eps = 0

# Set Original TV value. 
tv0 = 372273

#Read the Data. 
rmse = np.loadtxt(str(i)+'/rmse.txt')
cosAlph = np.loadtxt(str(i)+'/Cos_Alpha.txt')
tv = np.loadtxt(str(i)+'/tv.txt')
dd = np.loadtxt(str(i)+'/dd.txt')

print('Minimum Cosine Alpha value : ' +str(np.amin(cosAlph)))
print('Minimum dd value : ' +str(np.amin(dd)))
print('Minimum RMSE: ' + str(np.amin(rmse)))

x = np.arange(tv.shape[0]) + 1

fig, (ax1,ax2, ax3, ax4) = plt.subplots(4,1)
fig.subplots_adjust(hspace=1.0)

ax1.plot(x,rmse,color='m',linewidth=2.0)
ax1.set_ylabel('rmse')

ax2.plot(x,tv,color='green',linewidth=2.0)
ax2.set_ylabel('tv')
ax2.axhline(y=tv0, color='r')

ax3.plot(x,dd,color='black',linewidth=2.0)
ax3.axhline(y=eps, color='r')
ax3.set_ylabel('dd')

ax4.plot(x, cosAlph,color='blue',linewidth=2.0)
ax4.set_ylabel('Cosine Alpha')
ax4.set_xlabel('Number of Iterations')

plt.show()