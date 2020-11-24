from skimage import io
import scipy.ndimage
import numpy as np
import time, os
from tqdm import tqdm
import h5py

def load_data(vol_size, file_name):

    #sk-image loads tilt series as (z,y,x) so the axes need to be
    #swapped to return to (x,y,z)
    dir = 'Tilt_Series/'
    full_name = vol_size+'_'+file_name

    if full_name.endswith('.tiff'):
        tiltSeries = np.array(io.imread(dir+full_name), dtype=np.float32)
        tiltSeries = np.swapaxes(tiltSeries, 0, 2)
        ftype = '.tiff'
    elif full_name.endswith('.tif'):
        tiltSeries = np.array(io.imread(dir+full_name), dtype=np.float32)
        tiltSeries = np.swapaxes(tiltSeries, 0, 2)
        ftype = '.tif'
    elif full_name.endswith('.npy'):
        tiltSeries = np.load(dir+full_name)
        ftype = '.npy'

    # remove file type from name. 
    file_name = file_name.replace('_tiltser'+ftype, '')

    return (file_name,tiltSeries)


def load_h5_data(vol_size, file_name):
    dir = 'Tilt_Series/'
    if vol_size != '':
        full_name = vol_size+'_'+file_name
    else:
        full_name = file_name

    file = h5py.File(dir+full_name, 'r')
    vol = file['tiltSeries']
    angles = file['tiltAngles']

    file_name = file_name.replace('.h5', '')

    return (file_name, angles, vol)

def mpi_save_results(fname, tomo, saveRecon, meta=None, results=None):

    if tomo.rank() == 0: os.makedirs(fname[0], exist_ok=True)
    fullFname = '{}/{}.h5'.format(fname[0], fname[1])

    if saveRecon:
        tomo.save_recon(fullFname, 0)

    if tomo.rank() == 0:
        h5=h5py.File(fullFname, 'a')

        if meta != None:
            params = h5.create_group("parameters")
            for key,item in meta.items():
                params.attrs[key] = item

        if results != None:
            conv = h5.create_group("results")
            for key,item in results.items():
                conv.create_dataset(key, dtype=np.float32, data=item)

        h5.close()

def save_results(fname, meta=None, results=None):

    print('meta:', meta)
    print('results:', results)

    os.makedirs('results/'+fname[0]+'/', exist_ok=True)

    h5=h5py.File('results/{}/{}.h5'.format(fname[0],fname[1]), 'w')
    
    if meta != None:
        params = h5.create_group("parameters")
        for key,item in meta.items():
            params.attrs[key] = item

    if results != None:
        conv = h5.create_group("results")
        for key,item in results.items():
            conv.create_dataset(key, dtype=np.float32, data=item)

    h5.close()

def save_gif(fname, meta, gif):
    h5=h5py.File('results/{}/{}.h5'.format(fname[0],fname[1]), 'a')
    gif = h5.create_group("gif")
    gif.create_dataset("gif", dtype=np.float32, data=gif)
    gif.attrs["img_slice"] = meta

def save_recon(fname, meta, tomo):

    (Nslice, Nray, Nproj) = meta

    recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F') 
    for s in range(Nslice):
        recon[s,:,:] = tomo.get_recon(s)

    #print(np.amin(recon))
    #print(np.amax(recon))
 
    h5=h5py.File('results/{}/{}.h5'.format(fname[0],fname[1]), 'a')
    dset = h5.create_group("Reconstruction")
    dset.create_dataset("recon", dtype=np.float32, data=recon)
    dset.attrs["Nslice"] = Nslice
    dset.attrs["Nray"] = Nray
    dset.attrs["Nproj"] = Nproj

    h5.close()

def run(tomo, alg, beta=1, niter=1):
# Can Specify the Descent Parameter and nIter per slice.    

    if alg == 'SART':
        tomo.SART(beta,niter)
    elif alg == 'SIRT':
        tomo.SIRT(niter)
    elif alg == 'FBF':
        # Apply Positivity Always (?)
        tomo.FBF(True)

def initialize_algorithm(tomo, alg, initAlg=''):

    if alg == 'SART':
        #initAlg = {'random', 'sequential'}
        tomo.initialize_SART(initAlg)
    elif alg == 'SIRT':
        tomo.initialize_SIRT()
    elif alg == 'FBF':
        #initAlg = {'ram-lak', 'shepp-logan', 'hamming', etc.}
        tomo.initialize_FBF(initAlg)

    tomo.initialize_FP()

# Perform a basic functionality test for ASTRA and CUDA
def check_cuda():
    try:
        import astra
        import sys
        if not astra.use_cuda():
             print('No GPU support available')
             print('Please have a GPU to run these scripts')
             exit(1)
    except:
        print('Please have ASTRA installed!') 

