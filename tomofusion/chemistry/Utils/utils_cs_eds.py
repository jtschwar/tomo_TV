from scipy.sparse import csr_matrix
import numpy as np
import h5py, os

# Create Summation Matrix for Generating Synthetic HAADF Maps from Chemical Spectra.
def create_weighted_summation_matrix(nx, ny, nz, zNums, gamma,method=0):
	vals = np.zeros([nz*ny*nx], dtype=np.float16)
	row =  np.zeros([nz*ny*nx], dtype=int)
	col =  np.zeros([nz*ny*nx], dtype=int)
	vals[:] = 1

	ii = 0; ind = 0
	while ii < nz*nx*ny:
		for jj in range(nz):
			row[ii+jj] = ind
			col[ii+jj] = ind + nx*ny*jj
	
			if method == 0:
				pass 
			if method == 1:
				vals[ii+jj] = zNums[jj] / np.mean(zNums) # Z_{i}^{gamma} / ( sum_{i} ( Z_i ) / Nelements )
			if method == 2:
				vals[ii+jj] = zNums[jj]**gamma / np.mean(zNums**gamma) # Z_{i}^{gamma} / ( sum_{i} ( Z_i^{gamma} ) / Nelements )
			if method == 3:
				vals[ii+jj] = zNums[jj] / np.sum(zNums) # Z_{i} / sum_{i} Z_i
			if method == 4:
				vals[ii+jj] = zNums[jj]**gamma / np.sum(zNums**gamma) # Z_{i}^{gamma} / sum_{i} ( Z_i^{gamma} )

		ii += nz
		ind += 1
	A = csr_matrix((vals, (row, col)), shape=(nx*ny, nz*nx*ny), dtype=np.float32)

	return A

# Save the Multi-Element Tomograms in a H5 File with it's associated elemental tags.
def save_h5(dir, *nameStrings,**dataPairs):
    print(nameStrings)
    fileName = ''
    if len(nameStrings) == 0: fileName = 'temp'
    else:
        fileName = ''
        fileName = '_'.join(nameStrings)
    fileName = fileName + '.h5'
    filePath = os.path.join(dir,fileName)
    f = h5py.File(filePath, 'w')
    for key,item in dataPairs.items():
        print(key)
        f.create_dataset(key,data=item)
    f.close()
