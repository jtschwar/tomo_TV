from skimage import io
import scipy.ndimage
import numpy as np
import time, os
from tqdm import tqdm
import h5py

def parallelRay(Nside, angles, angleStart = 0):

    Nray = Nside

    pixelWidth = 1.0
    rayWidth = 1.0

    # Suppress warning messages that pops up when dividing zeros
    np.seterr(all='ignore')
    Nproj = angles.size # Number of projections

    # Ray coordinates at 0 degrees.
    offsets = np.linspace(-(Nray * 1.0 - 1) / 2,
                          (Nray * 1.0 - 1) / 2, Nray) * rayWidth
    # Intersection lines/grid Coordinates
    xgrid = np.linspace(-Nside * 0.5, Nside * 0.5, Nside + 1) * pixelWidth
    ygrid = np.linspace(-Nside * 0.5, Nside * 0.5, Nside + 1) * pixelWidth

    # Initialize vectors that contain matrix elements and corresponding
    # row/column numbers
    rows = np.zeros((2 * Nside * Nproj * Nray), dtype=np.float32)
    cols = np.zeros((2 * Nside * Nproj * Nray), dtype=np.float32)
    vals = np.zeros((2 * Nside * Nproj * Nray), dtype=np.float32)
    idxend = 0

    for i in tqdm(range(angleStart, Nproj)): # Loop over projection angles
        ang = angles[i] * np.pi / 180
        # Points passed by rays at current angles
        xrayRotated = np.cos(ang) * offsets
        yrayRotated = np.sin(ang) * offsets
        xrayRotated[np.abs(xrayRotated) < 1e-8] = 0
        yrayRotated[np.abs(yrayRotated) < 1e-8] = 0

        a = -np.sin(ang)
        a = rmepsilon(a)
        b = np.cos(ang)
        b = rmepsilon(b)

        for j in range(0, Nray): # Loop rays in current projection
            #Ray: y = tx * x + intercept
            t_xgrid = (xgrid - xrayRotated[j]) / a
            y_xgrid = b * t_xgrid + yrayRotated[j]

            t_ygrid = (ygrid - yrayRotated[j]) / b
            x_ygrid = a * t_ygrid + xrayRotated[j]
            # Collect all points
            t_grid = np.append(t_xgrid, t_ygrid)
            xx = np.append(xgrid, x_ygrid)
            yy = np.append(y_xgrid, ygrid)
            # Sort the coordinates according to intersection time
            I = np.argsort(t_grid)
            xx = xx[I]
            yy = yy[I]

            # Get rid of points that are outside the image grid
            Ix = np.logical_and(xx >= -Nside / 2.0 * pixelWidth,
                                xx <= Nside / 2.0 * pixelWidth)
            Iy = np.logical_and(yy >= -Nside / 2.0 * pixelWidth,
                                yy <= Nside / 2.0 * pixelWidth)
            I = np.logical_and(Ix, Iy)
            xx = xx[I]
            yy = yy[I]

            # If the ray pass through the image grid
            if (xx.size != 0 and yy.size != 0):
                # Get rid of double counted points
                I = np.logical_and(np.abs(np.diff(xx)) <=
                                   1e-8, np.abs(np.diff(yy)) <= 1e-8)
                I2 = np.zeros(I.size + 1)
                I2[0:-1] = I
                xx = xx[np.logical_not(I2)]
                yy = yy[np.logical_not(I2)]

                # Calculate the length within the cell
                length = np.sqrt(np.diff(xx)**2 + np.diff(yy)**2)
                #Count number of cells the ray passes through
                numvals = length.size

                # Remove the rays that are on the boundary of the box in the
                # top or to the right of the image grid
                check1 = np.logical_and(b == 0, np.absolute(
                    yrayRotated[j] - Nside / 2 * pixelWidth) < 1e-15)
                check2 = np.logical_and(a == 0, np.absolute(
                    xrayRotated[j] - Nside / 2 * pixelWidth) < 1e-15)
                check = np.logical_not(np.logical_or(check1, check2))

                if np.logical_and(numvals > 0, check):
                    # Calculate corresponding indices in measurement matrix
                    # First, calculate the mid points coord. between two
                    # adjacent grid points
                    midpoints_x = rmepsilon(0.5 * (xx[0:-1] + xx[1:]))
                    midpoints_y = rmepsilon(0.5 * (yy[0:-1] + yy[1:]))
                    #Calculate the pixel index for mid points
                    pixelIndicex = ((np.floor(Nside / 2.0 - midpoints_y / pixelWidth)) * # noqa TODO reformat this
                        Nside + (np.floor(midpoints_x /
                        pixelWidth + Nside / 2.0)))
                    # Create the indices to store the values to the measurement
                    # matrix
                    idxstart = idxend
                    idxend = idxstart + numvals
                    idx = np.arange(idxstart, idxend)
                    # Store row numbers, column numbers and values
                    rows[idx] = i * Nray + j
                    cols[idx] = pixelIndicex
                    vals[idx] = length
            else:
                print(("Ray No.", j + 1, "at", angles[i],
                       "degree is out of image grid!"))
    # Truncate excess zeros.
    rows = rows[:idxend]
    cols = cols[:idxend]
    vals = vals[:idxend]
    A = np.array([rows, cols, vals], dtype=np.float32, order='C')
    return A


def rmepsilon(input):
    if (input.size > 1):
        input[np.abs(input) < 1e-10] = 0
    else:
        if np.abs(input) < 1e-10:
            input = 0
    return input

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

def run(tomo, alg, beta=1):
    # Can Specify the Descent Parameter

    if alg == 'SIRT':
        tomo.SIRT(beta)
    elif alg == 'randART':
        tomo.randART(beta)
    else: 
        tomo.ART(beta)


def initialize_algorithm(tomo, alg, Nray, tiltAngles, angleStart = 0):

    print('Generating Measurement Matrix')
    A = parallelRay(Nray, tiltAngles, angleStart)

    if angleStart == 0: tomo.load_A(A)
    else: tomo.update_proj_angles(A, tiltAngles.shape[0])

    if alg == 'ART' or alg == 'randART':
        tomo.row_inner_product()

def create_projections(tomo, original_volume, SNR=0):

    # If creating simulation with noise, set background value to 1.
    if SNR != 0:
        original_volume[original_volume == 0] = 1

    # Load Volume and Collect Projections. 
    tomo.initialize_original_volume()
    Nslice = original_volume.shape[0]
    for s in range(Nslice):
        tomo.set_original_volume(original_volume[s,:,:],s)
    tomo.create_projections()

    # Apply poisson noise to volume.
    if SNR != 0:
        tomo.poisson_noise(SNR)

def load_exp_tilt_series(tomo, tiltSeries):
    (Nslice, Nray, Nproj) = tiltSeries.shape
    b = np.zeros([Nslice, Nray*Nproj])
    for s in range(Nslice):
        b[s,:] = tiltSeries[s,:,:].transpose().ravel()
    tomo.set_tilt_series(b)


def save_results(fname, meta, results, tomo, saveRecon = False):

    os.makedirs('results/'+fname[0]+'/', exist_ok=True)

    h5=h5py.File('results/{}/{}.h5'.format(fname[0],fname[1]), 'w')
    params = h5.create_group("parameters")
    for key,item in meta.items():
        params.attrs[key] = item

    conv = h5.create_group("results")
    for key,item in results.items():
        conv.create_dataset(key, dtype=np.float32, data=item)

    if saveRecon:

        Nslice, Nray = tomo.Nslice(), tomo.Nray()
        recon = np.zeros([Nslice, Nray, Nray], dtype=np.float32, order='F')
        for s in range(Nslice):
            recon[s,:,:] = tomo.get_recon(s)
        dset = h5.create_group("Reconstruction")
        dset.create_dataset("recon", dtype=np.float32, data=recon)
        dset.attrs["Nslice"] = Nslice
        dset.attrs["Nray"] = Nray
        # dset.attrs["Nproj"] = Nproj

    h5.close()

def mpi_save_results(fname, meta, results, tomo, saveREcon = False):

    if tomo.rank() == 0: os.makedirs(fname[0]+'/'+fname[1]+'/', exist_ok=True)
    fullFname = '{}/{}.h5'.format(fname[0], fname[1])

    if saveRecon:
        tomo.saveRecon(fullFname, 0)

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

def save_gif(fname, meta, gif):
    h5=h5py.File('Results/{}/{}.h5'.format(fname[0],fname[1]), 'a')
    gif = h5.create_group("gif")
    gif.create_dataset("gif", dtype=np.float32, data=gif)
    gif.attrs["img_slice"] = meta

