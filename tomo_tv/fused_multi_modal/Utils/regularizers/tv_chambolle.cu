//
//  tv_chambolle.cu
//
//  Created by Hovden Group on 9/1/20.
//  Copyright © 2020 Jonathan Schwartz. All rights reserved.
//

#include "tv_chambolle.h"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <cmath>
#include <stdio.h>

#define BLKXSIZE 8

#define MAX(x,y) (x>y?x:y)
#define MIN(x,y) (x<y?x:y)
#define ABS(x) (x>0?x:-x)

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )

//Measure Reconstruction's TV.
__global__ void tv_3D_kernel(float *vol, float *tv_recon, int nx, int ny, int nz)
{
    float eps = 1e-6;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (nx*ny)*k + i + nx*j;
    
    int ip = (nx*ny)*k + (i+1)%nx + nx*j;
    int jp = (nx*ny)*k + i + nx*((j+1)%ny);
    int kp = (nx*ny)*((k+1)%nz) + i + nx*j;

    if ((i < nx) && (j < ny) && (k < nz)) {
        tv_recon[ijk] = sqrt(eps + ( vol[ijk] - vol[ip] ) * ( vol[ijk] - vol[ip] )
                        + ( vol[ijk] - vol[jp] ) * ( vol[ijk] - vol[jp] )
                        + ( vol[ijk] - vol[kp] ) * ( vol[ijk] - vol[kp] ));
    }
    return;
}



__global__ void div_kernel(float *output, float *p1, float *p2, float *p3, int nx, int ny, int nz)
{
    int val1, val2, val3;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (nx*ny)*k + i + nx*j;

    if ((i < nx) && (j < ny) && (k < nz)) {

        // Boundary Conditions Along First Axis
        if i == 0 { val1 = p1[ijk]; }
        else if (i == nx-1) { val1 = -p1[ijk]; }
        else { val1 = p1[ijk] - p1[(nx*ny)*k + nx*j + (i-1)]; }

        // Boundary Conditions Along Second Axis
        if j == 0 { val2 = p2[ijk]; }
        else if (j == ny-1) { val2 = -p2[ijk]; }
        else { val2 = p2[ijk] - p2[(nx*ny)*k + nx*(j-1) + i]; }

        // Boundary Conditions Along Third Axis
        if k == 0 { val3 = p3[ijk]; }
        else if (k == nz-1) { val3 = -p3[ijk]; }
        else { val3 = p3[ijk] - p3[(nx*ny)*(k-1) + nx*j + i]; }

        output[ijk] = val1 + val2 + val3 - x[ijk]/lambda;
    }
}

__global__ void grad_kernel(float *D, float *P1, float *P2, float *P3, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (nx*ny)*k + i + nx*j;

    if ((i < nx) && (j < ny) && (k < nz)) {
        /* boundary conditions */
        if (i >= nx - 1) P1[ijk] = 0.0f; else P1[ijk] = D[index] - D[(nx*ny)*(k) + (i+1) + nx*j];
        if (j >= ny - 1) P2[ijk] = 0.0f; else P2[ijk] = D[index] - D[(nx*ny)*(k) + i + nx*(j+1)];
        if (k >= nz - 1) P3[ijk] = 0.0f; else P3[ijk] = D[index] - D[(nx*ny)*(k+1) + i + nx*j];
    }
}

__global__ void xi_update_kernel(float *gd1, float *gd2, float *gd3, float *P1, float *P2, float *P3, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (nx*ny)*k + i + nx*j;

    if ((i < nx) && (j < ny) && (k < nz)) {
        /* boundary conditions */
        if (i >= nx - 1) P1[ijk] = 0.0f; else P1[ijk] = D[index] - D[(nx*ny)*(k) + (i+1) + nx*j];
        if (j >= ny - 1) P2[ijk] = 0.0f; else P2[ijk] = D[index] - D[(nx*ny)*(k) + i + nx*(j+1)];
        if (k >= nz - 1) P3[ijk] = 0.0f; else P3[ijk] = D[index] - D[(nx*ny)*(k+1) + i + nx*j];
    }
}

__global__ void recon_update_kernel(float *vol, float *p1, float *p2, float *p3, float lambda)
{
    int val1, val2, val3;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (nx*ny)*k + i + nx*j;

    if ((i < nx) && (j < ny) && (k < nz)) {

        // Boundary Conditions Along First Axis
        if i == 0 { val1 = p1[ijk]; }
        else if (i == nx-1) { val1 = -p1[ijk]; }
        else { val1 = p1[ijk] - p1[(nx*ny)*k + nx*j + (i-1)]; }

        // Boundary Conditions Along Second Axis
        if j == 0 { val2 = p2[ijk]; }
        else if (j == ny-1) { val2 = -p2[ijk]; }
        else { val2 = p2[ijk] - p2[(nx*ny)*k + nx*(j-1) + i]; }

        // Boundary Conditions Along Third Axis
        if k == 0 { val3 = p3[ijk]; }
        else if (k == nz-1) { val3 = -p3[ijk]; }
        else { val3 = p3[ijk] - p3[(nx*ny)*(k-1) + nx*j + i]; }

        output[ijk] = recon[ijk] - (val1 + val2 + val3) * lambda;
    }
}

// MAIN HOST FUNCTION //
float cuda_tv_chambolle(float *recon, int ng, float lambda, int nx, int ny, int nz)
{
    // Operators for Global Reductions
    thrust::plus<float>  binary_op;
    square<float>        unary_op;

    // Initialize volume size and pointers. 
    int volSize = nx*ny*nz;
    float *d_recon, *tv_recon=NULL;
    float tv_gpu, tv_norm;
    float tau = 2/8;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));
   
    /*allocate space for images on device*/
    cudaMalloc((void**)&d_recon,volSize*sizeof(float));
    cudaMalloc((void**)&d_xi_1,volSize*sizeof(float));
    cudaMalloc((void**)&d_xi_2,volSize*sizeof(float));
    cudaMalloc((void**)&d_xi_3,volSize*sizeof(float));
    cudaMalloc((void**)&d_gdv, volSize*sizeof(float));

    // TV Recon is Always on Device. 
    cudaMemcpy(d_recon,recon,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_xi_1, 0.0f, volSize*sizeof(float));
    cudaMemset(d_xi_2, 0.0f, volSize*sizeof(float));
    cudaMemset(d_xi_3, 0.0f, volSize*sizeof(float));
    cudaMemset(d_gdv, 0.0f, volSize*sizeof(float));

    // Main Loop.
    for(int g=0; g < ng; g++)
    {   
        // Measure Isotropic TV - Gradient
        grad_kernel<<<dimGrid,dimBlock>>>(d_recon, d_xi_1, d_xi_2, d_xi_3, nx, ny, nz);
        cudaDeviceSynchronize();
        cudaPeekAtLastError();

        // Measure Norm of TV - Gradient
        thrust::device_vector<float> tv_vec(tv_recon, tv_recon + volSize);
        tv_norm = std::sqrt(thrust::transform_reduce(tv_vec.begin(), tv_vec.end(), unary_op, 0.0f, binary_op));
        cudaDeviceSynchronize();
        cudaPeekAtLastError();

        // TV Gradient Update 
        recon_update_kernel<<<dimGrid,dimBlock>>>(d_recon, tv_recon, tv_norm, dPOCS, nx, ny, nz);
        cudaDeviceSynchronize();
        cudaPeekAtLastError();
    }

    // Apply Positivity 
    positivity_kernel<<<dimGrid,dimBlock>>>(d_recon, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    // Measure TV
    tv_3D_kernel<<<dimGrid,dimBlock>>>(d_recon, tv_recon, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

     // Measure Norm of TV - Gradient
    thrust::device_vector<float> tv_vec(tv_recon, tv_recon + volSize);
    tv_gpu = thrust::reduce(tv_vec.begin(), tv_vec.end(), 0.0f, binary_op);

    // Copy Result Matrix from Device to Host Memory
    cudaMemcpy(recon, d_recon, volSize*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_recon);
    cudaFree(tv_recon);

    cudaDeviceSynchronize();

    return tv_gpu;
}
