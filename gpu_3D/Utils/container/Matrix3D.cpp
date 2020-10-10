#include "Matrix3D.h"
#include "matrix_ops.h"
//#include <Eigen/Core>

using namespace Eigen;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

Matrix3D::Matrix3D()
{
}

Matrix3D::Matrix3D(int Nx, int Ny, int Nz)
{
	nx = Nx;
	ny = Ny;
	nz = Nz;
	data = new float [nx*ny*nz];
	size = nx * ny * nz;
}

float Matrix3D::get_val(int i,int j,int k) {
	return data[(ny*nz)*i + ny*j + k];
}

int Matrix3D::index(int i, int j, int k){
    return (ny*nz)*i + ny*j + k;
}

float Matrix3D::sum() {
    return cuda_sum(data,nx,ny,nz);
}

float Matrix3D::norm() {
    return cuda_norm(data,nx,ny,nz);
}

void Matrix3D::positivity() {
    cuda_positivity(data,nx,ny,nz);
}

void Matrix3D::setData(Mat inBuffer, int slice)
{
    for (int yInd = 0; yInd < ny; yInd++) {
      for (int zInd = 0; zInd < nz; zInd++) {
          data[index(slice,yInd,zInd)] = inBuffer(yInd,zInd);
      }
    }
}

// Return Reconstruction to Python.
Mat Matrix3D::getData(int slice)
{
    Mat outBuffer(ny,nz);
    for (int yInd = 0; yInd < ny; yInd++) {
        for (int zInd = 0; zInd < nz; zInd++) {
            outBuffer(yInd,zInd) = data[index(slice,yInd,zInd)];
        }
    }
    return outBuffer;
}

