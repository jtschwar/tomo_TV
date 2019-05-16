import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

beta = np.loadtxt('beta.txt')
dPOCS = np.loadtxt('dPOCS.txt')
dd = np.loadtxt('dd.txt')
dg = np.loadtxt('dg.txt')
dp = np.loadtxt('dp.txt')
rmse = np.loadtxt('rmse.txt')
cosAlph = np.loadtxt('Cos_Alpha.txt')

print('Minimum Cosine Alpha value : ' +str(np.amin(cosAlph)))
print('Minimum dd value : ' +str(np.amin(dd)))

x = np.arange(beta.shape[0]) + 1

# plt.plot(x, dd)
# plt.axhline(y=1.0, color='r')
# plt.show()

fig, (ax1,ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7,1)
fig.subplots_adjust(hspace=1.0)

ax1.plot(x, beta, linewidth=2.0)
ax1.set_ylabel('beta')

ax2.plot(x,dPOCS,color='green',linewidth=2.0)
ax2.set_ylabel('dPOCS')

ax3.plot(x,dd,color='black',linewidth=2.0)
ax3.axhline(y=1.0, color='r')
ax3.set_ylabel('dd')

ax4.plot(x,dg,color='red',linewidth=2.0)
ax4.set_ylabel('dg')

ax5.plot(x,dp,color='green',linewidth=2.0)
ax5.set_ylabel('dp')

ax6.plot(x,rmse,color='green',linewidth=2.0)
ax6.set_ylabel('rmse')

ax7.plot(x,cosAlph,color='green',linewidth=2.0)
ax7.set_ylabel('Cosine Alpha')
ax7.set_xlabel('Number of Iterations')

plt.show()