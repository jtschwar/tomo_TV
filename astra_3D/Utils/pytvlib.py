from skimage import io
import scipy.ndimage
import numpy as np
import time, os
from tqdm import tqdm
import h5py

def timer(t0, counter, Niter):
    timeLeft = (time.time() - t0)/counter * (Niter - counter)
    timeLeftMin, timeLeftSec = divmod(timeLeft, 60)
    timeLeftHour, timeLeftMin = divmod(timeLeftMin, 60)
    print('Estimated time to complete: %02d:%02d:%02d' % (timeLeftHour, timeLeftMin, timeLeftSec))

def generate_tilt_series(volume, angles, num_tilts):

    Ny = volume.shape[1]
    Nz = volume.shape[2]
    N = volume.shape[1]

    # pad volume
    pad_y_pre = int(np.ceil((N - Ny) / 2.0))
    pad_y_post = int(np.floor((N - Ny) / 2.0))
    pad_z_pre = int(np.ceil((N - Nz) / 2.0))
    pad_z_post = int(np.floor((N - Nz) / 2.0))
    volume_pad = np.lib.pad(
        volume, ((0, 0), (pad_y_pre, pad_y_post), (pad_z_pre, pad_z_post)),
        'constant')

    Nslice = volume.shape[0]  # Number of slices along rotation axis.
    tiltSeries = np.empty([Nslice, N, num_tilts], dtype=float, order='F')

    for i in range(num_tilts):
        # Rotate volume about x-axis
        rotatedVolume = np.empty_like(volume_pad)
        scipy.ndimage.interpolation.rotate(
            volume_pad, angles[i], axes=(1, 2), reshape=False, order=1,
            output=rotatedVolume)
        # Calculate projection
        tiltSeries[:, :, i] = np.sum(rotatedVolume, axis=2)
    return tiltSeries

def load_data(vol_size, file_name):

    #sk-image loads tilt series as (z,y,x) so the axes need to be
    #swapped to return to (x,y,z)
    dir = 'Tilt_Series/'
    full_name = vol_size+file_name

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
    full_name = vol_size+file_name

    file = h5py.File(dir+full_name, 'r')
    vol = file['tiltSeries']
    angles = file['tiltAngles']

    file_name = file_name.replace('.h5', '')

    return (file_name, angles, vol)

def save_results(fname, meta, results):

    os.makedirs('results/'+fname[0]+'/', exist_ok=True)

    h5=h5py.File('results/{}/{}.h5'.format(fname[0],fname[1]), 'w')
    params = h5.create_group("parameters")
    for key,item in meta.items():
        params.attrs[key] = item

    conv = h5.create_group("convergence")
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
        recon[s,:,:] = tomo.getRecon(s)

    print(np.amin(recon))
    print(np.amax(recon))
 
    h5=h5py.File('results/{}/{}.h5'.format(fname[0],fname[1]), 'a')
    dset = h5.create_group("Reconstruction")
    dset.create_dataset("recon", dtype=np.float32, data=recon)
    dset.attrs["Nslice"] = Nslice
    dset.attrs["Nray"] = Nray
    dset.attrs["Nproj"] = Nproj

    h5.close()

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


    
