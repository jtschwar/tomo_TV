from matplotlib import pyplot as plt
import numpy as np

def ART_results(dd):
    x = np.arange(dd.shape[0]) + 1
    plt.plot(x,dd,color='black', linewidth=2.0)
    plt.xlabel('Number of Iterations')
    plt.title('Final dd: ' +str(round(dd[-1],2)), loc='right', fontsize=10)
    plt.title('DD', loc='left', fontweight='bold')
    plt.show()

def live_ART_results(dd, i):
    x = np.arange(i) + 1
    plt.plot(x,dd[:i], color='black', linewidth=2.0)
    plt.xlabel('Number of Iterations')
    plt.title('Final dd: ' +str(round(dd[i],2)), loc='right', fontsize=10)
    plt.title('DD', loc='left', fontweight='bold')
    plt.draw()
    plt.pause(0.001)

def SIRT_results(dd):
    x = np.arange(dd.shape[0]) + 1
    plt.plot(x,dd)
    plt.xlabel('Number of Iterations')
    plt.title('DD', fontweight='bold')
    plt.show()

def ASD_results(dd, eps, tv):
    x = np.arange(tv.shape[0]) + 1

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,4))
    fig.subplots_adjust(hspace=0.4)

    ax1.plot(x, tv,color='blue', linewidth=2.0)
    ax1.set_title('Final TV: ' +str(round(tv[-1],2)), loc='right', fontsize=10)
    ax1.set_title('TV', loc='center', fontweight='bold')
    ax1.set_xticklabels([])

    ax2.plot(x,dd,color='black', linewidth=2.0)
    ax2.axhline(y=eps, color='r')
    ax2.set_title('Final dd: ' +str(round(dd[-1],2)), loc='right', fontsize=10)
    ax2.set_title('DD', loc='left', fontweight='bold')
    ax2.set_xlabel('Number of Iterations', fontweight='bold')

    plt.show()

def ASD_live_plot(dd, eps, tv, i):

    x = np.arange(i) + 1

    ax1 = plt.subplot(211, frameon=True)
    ax1.plot(x, tv[:i],color='blue', linewidth=2.0)
    ax1.set_title('Final TV: ' +str(round(tv[i],2)), loc='right', fontsize=10)
    ax1.set_title('TV', loc='center', fontweight='bold')
    ax1.set_xticklabels([])

    ax2 = plt.subplot(212, frameon=True)
    ax2.plot(x,dd[:i],color='black', linewidth=2.0)
    ax2.axhline(y=eps, color='r')
    ax2.set_title('Final dd: ' +str(round(dd[i],2)), loc='right', fontsize=10)
    ax2.set_title('DD', loc='left', fontweight='bold')
    ax2.set_xlabel('Number of Iterations', fontweight='bold')

    plt.draw()
    plt.pause(0.001)
    plt.clf()

def exp_time_tv_live_plot(dd, eps, tv, Niter, i):
    
    #Plot DD and TV vs Total Number of Projections.

    xiter = np.arange(i) + 1

    dd_proj = np.zeros(i)
    tv_proj = np.zeros(i)

    for i in range(xiter.shape[0]):
        dd_proj[i] = dd[np.sum(Niter[:i])]
        tv_proj[i] = tv[np.sum(Niter[:i])]

    ax1 = plt.subplot(211, frameon=True)
    ax1.plot(xiter, tv_proj,color='blue', linewidth=2.0)
    ax1.set_title('Final TV: ' +str(round(tv_proj[i],2)), loc='right', fontsize=10)
    ax1.set_title('TV', loc='center', fontweight='bold')
    ax1.set_xticklabels([])

    ax2 = plt.subplot(212, frameon=True)
    ax2.plot(xiter,dd_proj,color='black', linewidth=2.0)
    ax2.axhline(y=eps, color='r')
    ax2.set_title('Final dd: ' +str(round(dd_proj[i],2)), loc='right', fontsize=10)
    ax2.set_title('DD', loc='left', fontweight='bold')
    ax2.set_xlabel('Number of Projections', fontweight='bold')

    plt.draw()
    plt.pause(0.001)
    plt.clf()

def sim_time_tv_live_plot(dd,eps, tv, tv0, rmse, Niter, i):

    #Plot DD and TV vs Total Number of Projections.

    xiter = np.arange(Niter.shape[0]) + 1

    dd_proj =  np.zeros(Niter.shape[0]) + 1
    tv_proj =  np.zeros(Niter.shape[0]) + 1
    rmse_proj = np.zeros(Niter.shape[0]) + 1

    for i in range(xiter.shape[0]):
        dd_proj[i] = dd[np.sum(Niter[:i])]
        tv_proj[i] = tv[np.sum(Niter[:i])]
        rmse_proj[i] = rmse[np.sum(Niter[:i])]

    # fig.subplots_adjust(hspace=0.7)

    ax1 = plt.subplot(311, frameon=True)
    ax1.plot(x, tv,color='blue', linewidth=2.0)
    ax1.axhline(y=tv0, color='r')
    ax1.set_title('Final TV: ' +str(round(tv[-1],2)), loc='right', fontsize=10)
    ax1.set_title('TV', loc='center', fontweight='bold')
    ax1.set_xticklabels([])

    ax2 = plt.subplot(312, frameon=True)
    ax2.plot(x,dd,color='black', linewidth=2.0)
    ax2.axhline(y=eps, color='r')
    ax2.set_title('Final dd: ' +str(round(dd[-1],2)), loc='right', fontsize=10)
    ax2.set_title('DD', loc='left', fontweight='bold')
    ax2.set_xticklabels([])

    ax3 = plt.subplot(313, frameon=True)
    ax3.plot(x, rmse, color='m', linewidth=2.0)
    ax3.set_title('Final rmse: ' +str(round(rmse[-1],2)), loc='right', fontsize=10)
    ax3.set_title('RMSE', loc='left', fontweight='bold')
    ax3.set_xlabel('Number of Iterations', fontweight='bold')

    plt.draw()
    plt.pause(0.001)
    plt.clf()
