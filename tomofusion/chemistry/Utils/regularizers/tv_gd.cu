/* CUDA implementation of GD-TV [1] denoising/regularization model (3D case)
 *
 * Input Parameters:
 * 1. Noisy image/volume
 * 2. lambdaPar - regularization parameter
 * 3. Number of iterations
 * 4. eplsilon: tolerance constant
 * 5. TV-type: methodTV - 'iso' (0) or 'l1' (1)
 * 6. nonneg: 'nonnegativity (0 is OFF by default)
 *
 * Output:
 * [1] Filtered/regularized image/volume
 * [2] Information vector which contains [iteration no., reached tolerance]
 *
 * This function is based on the Matlab's code and paper by
 * [1] Amir Beck and Marc Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems"
 */
//
//  tv_gd.cu
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "tv_gd.h"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <cmath>
#include <stdio.h>

// What's the best block size? 8? 16? How can we calculate this? 
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

    int ijk = (ny*nz)*i + nz*j + k;
    
    int ip = (ny*nz)*((i+1)%nx) + nz*j + k;
    int jp = (ny*nz)*i + nz*((j+1)%ny) + k;
    int kp = (ny*nz)*i + nz*j + ((k+1)%nz);

    if ((i < nx) && (j < ny) && (k < nz)) {
        tv_recon[ijk] = sqrt(eps + ( vol[ijk] - vol[ip] ) * ( vol[ijk] - vol[ip] )
                        + ( vol[ijk] - vol[jp] ) * ( vol[ijk] - vol[jp] )
                        + ( vol[ijk] - vol[kp] ) * ( vol[ijk] - vol[kp] ));
    }
    return;
}



// Gradient Descent
__global__ void tv_gradient_3D_kernel(float *recon, float *tv_recon, int nx, int ny, int nz)
{
    float eps = 1e-6;
    float v1n, v2n, v3n, v4n, v1d, v2d, v3d, v4d;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (ny*nz)*i + nz*j + k;

    int ip = (ny*nz)*((i+1)%nx) + nz*j + k;
    int im = (ny*nz)*((i-1+nx)%nx) + nz*j + k;

    int jp = (ny*nz)*i + nz*((j+1)%ny) + k;
    int jm = (ny*nz)*i + nz*((j-1+ny)%ny) + k;

    int kp = (ny*nz)*i + nz*j + ((k+1)%nz);
    int km = (ny*nz)*i + nz*j + ((k-1+nz)%nz);

    int im_jp = (ny*nz)*((i-1+nx)%nx) + nz*((j+1)%ny) + k;
    int ip_jm = (ny*nz)*((i+nx)%nx) + nz*((j-1+ny)%ny) + k;

    int jm_kp = (ny*nz)*i + nz*((j-1+ny)%ny) + ((k+1)%nz);
    int jp_km = (ny*nz)*i + nz*((j+1)%ny) + ((k-1-nz)%nz);

    int im_kp = (ny*nz)*((i-1+nx)%nx) + nz*j + ((k+1)%nz);
    int ip_km = (ny*nz)*((i+1)%nx) + nz*j + ((k-1-nz)%nz);    

    if ((i < nx) && (j < ny) && (k < nz)) {

        v1n = 3.0*recon[ijk] - recon[ip] - recon[jp] - recon[kp];
        v1d = sqrt(eps + ( recon[ijk] - recon[ip] ) * ( recon[ijk] - recon[ip] )
                          +  ( recon[ijk] - recon[jp] ) * ( recon[ijk] - recon[jp] )
                          +  ( recon[ijk] - recon[kp] ) * ( recon[ijk] - recon[kp] ));

        v2n = recon[ijk] - recon[im];
        v2d = sqrt(eps + ( recon[im] - recon[ijk] ) * ( recon[im] - recon[ijk] )
                          +  ( recon[im] - recon[im_jp] ) * ( recon[im] - recon[im_jp] )
                          +  ( recon[im] - recon[im_kp] ) * ( recon[im] - recon[im_kp] ));

        v3n = recon[ijk] - recon[jm];
        v3d = sqrt(eps + ( recon[jm] - recon[ip_jm] ) * ( recon[jm] - recon[ip_jm] )
                          +  ( recon[jm] - recon[ijk] ) * ( recon[jm] - recon[ijk] )
                          +  ( recon[jm] - recon[jm_kp] ) * ( recon[jm] - recon[jm_kp] ) );
        
        v4n = recon[ijk] - recon[km];
        v4d = sqrt(eps + ( recon[km] - recon[ip_km] ) * ( recon[km] - recon[ip_km] )
                          + ( recon[km] - recon[jp_km] ) * ( recon[km] - recon[jp_km] )
                          + ( recon[km] - recon[ijk] ) * ( recon[km] - recon[ijk] ) );
        
        tv_recon[ijk] = v1n/v1d + v2n/v2d + v3n/v3d + v4n/v4d;
    }
    return;
}

__global__ void tv_gradient_update_3D_kernel(float *recon, float *tv_recon, float tv_norm, float dPOCS, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (ny*nz)*i + nz*j + k;

    if ((i < nx) && (j < ny) && (k < nz)) {

        recon[ijk] -= dPOCS * tv_recon[ijk] / tv_norm;

    }
    return;
}

// Main Host Function
__global__ void positivity_kernel(float *recon, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (ny*nz)*i + nz*j + k;

    if ((i < nx) && (j < ny) && (k < nz)) {
        if (recon[ijk] < 0.0f) {
            recon[ijk] = 0.0f; }
        }
    return;
}

float cuda_tv_3D(float *recon, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();

        // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

    // Operators for Global Reductions
    thrust::plus<float>  binary_op;

    // Initialize volume size and pointers. 
    int volSize = nx*ny*nz;
    float *d_recon, *tv_recon=NULL;
    float tv_gpu;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));
   
    /*allocate space for images on device*/
    cudaMalloc((void**)&d_recon,volSize*sizeof(float));
    cudaMalloc((void**)&tv_recon,volSize*sizeof(float));

    // TV Recon is Always on Device. 
    cudaMemcpy(d_recon,recon,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(tv_recon, 0.0f, volSize*sizeof(float));

    // Measure TV
    tv_3D_kernel<<<dimGrid,dimBlock>>>(d_recon, tv_recon, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

     // Measure Norm of TV - Gradient
    thrust::device_vector<float> tv_vec(tv_recon, tv_recon + volSize);
    tv_gpu = thrust::reduce(tv_vec.begin(), tv_vec.end(), 0.0f, binary_op);

    cudaFree(d_recon);
    cudaFree(tv_recon);

    cudaDeviceSynchronize();

    return tv_gpu;
}

float cuda_tv_gd_4D(float *vol, int ng, float dPOCS, int nx, int ny, int nz, int Nel, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();

        // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

    // Operators for Global Reductions
    thrust::plus<float>  binary_op;
    square<float>        unary_op;

    // Initialize volume size and pointers. 
    int volSize = nx*ny*nz;
    float *d_recon, *tv_recon=NULL;
    float tv_gpu =0, tv_norm=0;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));
   
    /*allocate space for images on device*/
    cudaMalloc((void**)&d_recon,volSize*sizeof(float));
    cudaMalloc((void**)&tv_recon,volSize*sizeof(float));

    for (int e=0; e<Nel; e++) 
    {
        // TV Recon is Always on Device. 
        cudaMemcpy(d_recon,&vol[e*volSize],volSize*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemset(tv_recon, 0, volSize*sizeof(float));

        // Measure TV
        tv_3D_kernel<<<dimGrid,dimBlock>>>(d_recon, tv_recon, nx, ny, nz);
        cudaDeviceSynchronize();
        cudaPeekAtLastError();

        // Measure Norm of TV - Gradient
        thrust::device_vector<float> tv_vec(tv_recon, tv_recon + volSize);
        tv_gpu += thrust::reduce(tv_vec.begin(), tv_vec.end(), 0.0f, binary_op);

        // Main Loop.
        for(int g=0; g < ng; g++)
        {   
            // Measure Isotropic TV - Gradient
            tv_gradient_3D_kernel<<<dimGrid,dimBlock>>>(d_recon, tv_recon, nx, ny, nz);
            cudaDeviceSynchronize();
            cudaPeekAtLastError();

            // Measure Norm of TV - Gradient
            thrust::device_vector<float> tv_vec(tv_recon, tv_recon + volSize);
            tv_norm = std::sqrt(thrust::transform_reduce(tv_vec.begin(), tv_vec.end(), unary_op, 0.0f, binary_op));
            cudaDeviceSynchronize();
            cudaPeekAtLastError();

            // TV Gradient Update 
            tv_gradient_update_3D_kernel<<<dimGrid,dimBlock>>>(d_recon, tv_recon, tv_norm, dPOCS, nx, ny, nz);
            cudaDeviceSynchronize();
            cudaPeekAtLastError();
        }

        // Apply Positivity 
        positivity_kernel<<<dimGrid,dimBlock>>>(d_recon, nx, ny, nz);
        cudaDeviceSynchronize();
        cudaPeekAtLastError();

        // Copy Result Matrix from Device to Host Memory
        cudaMemcpy(&vol[e*volSize], d_recon, volSize*sizeof(float), cudaMemcpyDeviceToHost);  
    }
    cudaFree(d_recon);
    cudaFree(tv_recon);

    cudaDeviceSynchronize();

    return tv_gpu;
}