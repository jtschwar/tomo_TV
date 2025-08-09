#include "Matrix4D.h"
#include "matrix_ops.h"
#include <Eigen/Core>

using namespace Eigen;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;

Matrix4D::Matrix4D()
{
}

Matrix4D::Matrix4D(int Nel, int Nx, int Ny, int Nz) {
	nx = Nx;
	ny = Ny;
	nz = Nz;
    nel = Nel;
	data = new float [nx*ny*nz*Nel];
	volSize = nx * ny * nz;
}

// for (int e=0; e<nel; e++) {
    // for (int i=0; i<nx; i++) {
        // for (int j=0; j<ny; j++) {
            // for (int k=0; k<nz; k++)
        // }
    // }
// }

float Matrix4D::get_val(int e, int i, int j, int k) {
	return data[e*(nx*ny*nz) + (ny*nz)*i + nz*j + k];
}

int Matrix4D::index(int e, int i, int j, int k){
    return e*(nx*ny*nz) + (ny*nz)*i + nz*j + k;
}

void Matrix4D::positivity() {
    cuda_positivity_4D(data,nx,ny,nz,nel);
}

void Matrix4D::setData2D(Mat inBuffer, int element, int slice) {
    for (int yInd = 0; yInd < ny; yInd++) {
      for (int zInd = 0; zInd < nz; zInd++) {
          data[index(element,slice,yInd,zInd)] = inBuffer(yInd,zInd);
      }
    }
}

void Matrix4D::setData1D(Vec inBuffer, int element, int slice) {
    int ind = 0;
    for (int yInd = 0; yInd < ny; yInd++) {
      for (int zInd = 0; zInd < nz; zInd++) {
          ind = zInd + yInd * nz;
          data[index(element,slice,yInd,zInd)] = inBuffer(ind);
      }
    }
}

// Return Reconstruction to Python.
Mat Matrix4D::getData2D(int element, int slice) {
    Mat outBuffer(ny,nz);
    for (int yInd = 0; yInd < ny; yInd++) {
        for (int zInd = 0; zInd < nz; zInd++) {
            outBuffer(yInd,zInd) = data[index(element,slice,yInd,zInd)];
        }
    }
    return outBuffer;
}

// Vec Matrix4D::getData1D(int element, int slice)
// {
//     int ind = 0;
//     Vec outBuffer(ny*nz);
//     for (int yInd = 0; yInd < ny; yInd++) {
//         for (int zInd = 0; zInd < nz; zInd++) {
//             ind = zInd + yInd * nz;
//             outBuffer[ind] = data[index(element,slice,yInd,zInd)];
//         }
//     }
//     return outBuffer;
// }


float *Matrix4D::getData1D(int element, int slice) {
    int ind = 0;
    // Vec outBuffer(ny*nz);
    float *outBuffer[ny*nz];
    for (int yInd = 0; yInd < ny; yInd++) {
        for (int zInd = 0; zInd < nz; zInd++) {
            ind = zInd + yInd * nz;
            outBuffer[ind] = &data[index(element,slice,yInd,zInd)];
        }
    }
    return *outBuffer;
}