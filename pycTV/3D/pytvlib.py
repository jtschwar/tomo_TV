import numpy as np

def tv_derivative(recon):
    r = np.lib.pad(recon, ((1, 1), (1, 1), (1, 1)), 'edge')
    v1n = 3 * r - np.roll(r, 1, axis=0) - \
                          np.roll(r, 1, axis=1) - np.roll(r, 1, axis=2) # noqa TODO reformat this
    v1d = np.sqrt(1e-8 + (r - np.roll(r, 1, axis=0))**2 + (r -
                  np.roll(r, 1, axis=1))**2 + (r - np.roll(r, 1, axis=2))**2) # noqa TODO reformat this

    v2n = r - np.roll(r, -1, axis=0)
    v2d = np.sqrt(1e-8 + (np.roll(r, -1, axis=0) - r)**2 +
            (np.roll(r, -1, axis=0) -  # noqa TODO reformat this
             np.roll(np.roll(r, -1, axis=0), 1, axis=1))**2 +
            (np.roll(r, -1, axis=0) - np.roll(np.roll(r, -1, axis=0), 1, axis=2))**2) # noqa TODO reformat this

    v3n = r - np.roll(r, -1, axis=1)
    v3d = np.sqrt(1e-8 + (np.roll(r, -1, axis=1) - np.roll(np.roll(r, -1, axis=1), 1, axis=0))**2 + # noqa TODO reformat this
                  (np.roll(r, -1, axis=1) - r)**2 + # noqa TODO reformat this
                  (np.roll(r, -1, axis=1) - np.roll(np.roll(r, -1, axis=1), 1, axis=2))**2) # noqa TODO reformat this

    v4n = r - np.roll(r, -1, axis=2)
    v4d = np.sqrt(1e-8 + (np.roll(r, -1, axis=2) - np.roll(np.roll(r, -1, axis=2), 1, axis=0))**2 + # noqa TODO reformat this
                  (np.roll(r, -1, axis=2) -  # noqa TODO reformat this
                  np.roll(np.roll(r, -1, axis=2), 1, axis=1))**2 +
                  (np.roll(r, -1, axis=2) - r)**2) # noqa TODO reformat this

    v = v1n / v1d + v2n / v2d + v3n / v3d + v4n / v4d
    v = v[1:-1, 1:-1, 1:-1]
    return v


def tv(recon):
    r = np.lib.pad(recon, ((1, 1), (1, 1), (1, 1)), 'edge')
    tv = np.sqrt(1e-8 + (r - np.roll(r, 1, axis=0))**2 +
                 (r - np.roll(r, 1, axis=1))**2 +
                 (r - np.roll(r, 1, axis=2))**2)
    tv = np.sum(tv[1:-1, 1:-1, 1:-1])
    return tv

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
