import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

#Theta Max Values. 
index = [20, 40, 60, 80, 100, 120, 140, 160, 180]

# Set Epsilon value. 
eps = 0

# Set the original TV value. 
tv0 = 372273

for j in range(len(index)):

	i = index[j]

	#Read the Data. 
	rmse = np.loadtxt(str(i)+'/rmse.txt')
	cosAlph = np.loadtxt(str(i)+'/Cos_Alpha.txt')
	tv = np.loadtxt(str(i)+'/tv.txt')
	dd = np.loadtxt(str(i)+'/dd.txt')

	x = np.arange(tv.shape[0]) + 1

	fig, (ax1,ax2, ax3, ax4) = plt.subplots(4,1, figsize=(7, 7))
	fig.subplots_adjust(hspace=1.0)

	ax1.plot(x,rmse,color='m',linewidth=2.0)
	ax1.set_ylabel('RMSE', fontweight='bold')
	ax1.set_title('Min RMSE: ' +str(np.amin(rmse)), loc='right', fontsize=10)

	ax2.plot(x,tv,color='green',linewidth=2.0)
	ax2.set_ylabel('TV' ,fontweight='bold')
	ax2.axhline(y=tv0, color='r')
	ax2.set_title('Final TV: ' +str(tv[-1]), loc='right', fontsize=10)

	ax3.plot(x,dd,color='black', linewidth=2.0)
	ax3.axhline(y=eps, color='r')
	ax3.set_ylabel('DD', fontweight='bold')
	ax3.set_title('Min dd: ' +str(np.amin(dd)), loc='right', fontsize=10)

	ax4.plot(x, cosAlph,color='blue',linewidth=2.0)
	ax4.set_ylabel('Cosine Alpha', fontweight='bold')
	ax4.set_xlabel('Number of Iterations', fontweight='bold')
	ax4.set_title('Min Cosine-Alpha: ' +str(np.amin(cosAlph)), loc='right', fontsize=10)

	plt.savefig(str(i)+'/plot.png')