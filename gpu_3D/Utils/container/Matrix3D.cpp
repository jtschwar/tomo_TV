#include "Matrix3D.h"
#include "matrix_ops.h"

using namespace Eigen;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

// Constructors
Matrix3D::Matrix3D() { }
Matrix3D::Matrix3D(int Nx, int Ny, int Nz) {
	nx = Nx;
	ny = Ny;
	nz = Nz;
	data = new float [nx*ny*nz];
	size = nx * ny * nz; }

// Get Scalar Value
float Matrix3D::get_val(int i,int j,int k) { return data[(ny*nz)*i + ny*j + k]; }

// Get Index from (i,j,k) 
int Matrix3D::index(int i, int j, int k) {
    return (ny*nz)*i + nz*j + k;
//    return i + nx*j + (nx*ny)*k;
}

void Matrix3D::setData(Mat inBuffer, int slice) {
    for (int yInd = 0; yInd < ny; yInd++) {
      for (int zInd = 0; zInd < nz; zInd++) {
            data[index(slice,yInd,zInd)] = inBuffer(yInd,zInd);
      }
    }   
}

// Return Reconstruction to Python.
Mat Matrix3D::getData(int slice) {
    Mat outBuffer(ny,nz);
    for (int yInd = 0; yInd < ny; yInd++) {
        for (int zInd = 0; zInd < nz; zInd++) {
            outBuffer(yInd,zInd) = data[index(slice,yInd,zInd)];
        }
    }
    return outBuffer; 
}

// Sum all Values in Reconstruction
float Matrix3D::sum() { return cuda_sum(data,nx,ny,nz); }

// L1 Norm
float Matrix3D::l1_norm() { return cuda_l1_norm(data,nx,ny,nz); }

// L2 Norm
float Matrix3D::norm() { return cuda_norm(data,nx,ny,nz); }

// Remove Negative Voxels from Volume
void Matrix3D::positivity() { cuda_positivity(data,nx,ny,nz); }

// Set Values Equal to Value Equal to Background Value
void Matrix3D::setBackground(int backgroundValue) { cuda_set_background(data,backgroundValue,nx,ny,nz); }

// Soft Thresholding Operation
void Matrix3D::soft_threshold(float lambda) { cuda_soft_threshold(data,lambda,nx,ny,nz); }

