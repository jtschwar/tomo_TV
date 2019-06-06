import numpy as np

def tv_derivative(recon):
    #r = np.lib.pad(recon, ((1, 1), (1, 1)), 'edge')
    r = np.lib.pad(recon, ((1, 1), (1, 1)), 'constant', constant_values = 0)
    v1n = 4 * r - 2 * np.roll(r, 1, axis=0) - \
                          2 * np.roll(r, 1, axis=1)  # noqa TODO reformat this
    v1d = np.sqrt(1e-8 + (r - np.roll(r, 1, axis=0))**2 + 
                  (r - np.roll(r, 1, axis=1))**2 ) # noqa TODO reformat this

    v2n = 2*(r - np.roll(r, -1, axis=0))
    v2d = np.sqrt(1e-8 + (np.roll(r, -1, axis=0) - r)**2 +
            (np.roll(r, -1, axis=0) -  # noqa TODO reformat this
             np.roll(np.roll(r, -1, axis=0), 1, axis=1))**2) # noqa TODO reformat this

    v3n = 2*(r - np.roll(r, -1, axis=1))
    v3d = np.sqrt(1e-8 + (np.roll(r, -1, axis=1) - np.roll(np.roll(r, -1, axis=1), 1, axis=0))**2 + # noqa TODO reformat this
                  (np.roll(r, -1, axis=1) - r)**2) # noqa TODO reformat this

    v = v1n / v1d + v2n / v2d + v3n / v3d 
    v = v[1:-1, 1:-1]
    v = v / np.linalg.norm(v)
    return v

def tv(recon):
    #r = np.lib.pad(recon, ((1, 1), (1, 1)), 'edge')
    r = np.lib.pad(recon, ((1, 1), (1, 1)), 'constant', constant_values=0)
    tv = np.sqrt((r - np.roll(r, 1, axis=0))**2 +
                 (r - np.roll(r, 1, axis=1))**2 )
    tv = np.sum(tv[1:-1, 1:-1])
    return tv
