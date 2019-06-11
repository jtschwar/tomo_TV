import matplotlib.pyplot as plt
import numpy as np

# Epsilon Values. 
eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
final_tv = np.zeros(len(eps))

for i in range(len(eps)):

	# #Read the Data. 
	tv = np.load('Epsilon_Test/' + str(eps[i])+'/tv.npy')
	dd = np.load('Epsilon_Test/' + str(eps[i])+'/dd.npy')

	print('Iteration: ' + str(i) + ', Min TV: ' + str(np.amin(tv)))

	final_tv[i] = tv[-1]

	x = np.arange(tv.shape[0]) + 1

	fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,4))
	fig.subplots_adjust(hspace=0.4)

	ax1.plot(x, tv,color='blue', linewidth=2.0)
	ax1.set_title('Min TV: ' +str(np.amin(tv)), loc='right', fontsize=10)
	ax1.set_title('TV', loc='center', fontweight='bold')
	ax1.set_xticklabels([])

	ax2.plot(x,dd,color='black', linewidth=2.0)
	ax2.axhline(y=eps[i], color='r')
	ax2.set_title('Min dd: ' +str(np.amin(dd)), loc='right', fontsize=10)
	ax2.set_title('DD', loc='center', fontweight='bold')
	ax2.set_xlabel('Number of Iterations', fontweight='bold')

	plt.savefig('Epsilon_Test/' + str(eps[i])+'/Individual_plot.png')

plt.figure(figsize=(8,4))
plt.plot(eps, final_tv, linewidth=2.0)
plt.title('TV', fontweight='bold')
plt.title('Min TV: ' +str(np.amin(final_tv)), loc='right', fontsize=10)
plt.xlabel('Epsilon', fontweight='bold')
plt.xticks(np.arange(min(eps), max(eps), 0.2))

plt.savefig('Epsilon_Test/plot.png')
