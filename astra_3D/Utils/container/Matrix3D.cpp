#include "Matrix3D.h"
#include "matrix_ops.h"

Matrix3D::Matrix3D()
{
}

Matrix3D::Matrix3D(int Nx, int Ny, int Nz)
{
	nx = Nx;
	ny = Ny;
	nz = Nz;
	data = new float [nx*ny*nz];
	size = nx * ny *nz;
}

float Matrix3D::get_val(int i,int j,int k) {
	return data[(ny*nz)*i + ny*j + k];
}

int Matrix3D::calc_index(int i, int j, int k){
    return (ny*nz)*i + ny*j + k;
}

float Matrix3D::sum() {
    return cuda_sum(data,nx,ny,nz);
}

float Matrix3D::norm() {
    return cuda_norm(data,nx,ny,nz);
}

