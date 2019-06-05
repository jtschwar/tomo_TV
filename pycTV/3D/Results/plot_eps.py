import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections
from skimage import io
import os.path

#Theta Max Values. 
eps = np.linspace(0.5, 1.5, 11)
eps = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

for i in range(len(eps)):

	#Read the Data. 
	tv = np.load('Epsilon_Test/' + str(eps[i])+'/tv.npy')
	dd = np.load('Epsilon_Test/' + str(eps[i])+'/dd.npy')

	x = np.arange(tv.shape[0]) + 1

	fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,4))
	fig.subplots_adjust(hspace=0.4)

	ax1.plot(x, tv,color='blue', linewidth=2.0)
	ax1.set_title('Min TV: ' +str(np.amin(tv)), loc='right', fontsize=10)
	ax1.set_title('TV', loc='center', fontweight='bold')
	ax1.set_xticklabels([])

	ax2.plot(x,dd,color='black', linewidth=2.0)
	ax2.axhline(y=eps[0], color='r')
	ax2.set_title('Min dd: ' +str(np.amin(dd)), loc='right', fontsize=10)
	ax2.set_title('DD', loc='center', fontweight='bold')
	ax2.set_xlabel('Number of Iterations', fontweight='bold')

	plt.savefig('Epsilon_Test/' + str(eps[i])+'/plot.png')


