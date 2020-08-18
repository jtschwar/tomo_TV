//
//  matrix_ops.cu
//
//  Created by Hovden Group on 8/17/2020.
//  Copyright © 2020 Jonathan Schwartz. All rights reserved.
//

#include "matrix_ops.h"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <cmath>

// What's the best block size? 8? 16? How can we calculate this? 
#define BLKXSIZE 8

#define MAX(x,y) (x>y?x:y)
#define MIN(x,y) (x<y?x:y)
#define ABS(x) (x>0?x:-x)

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )

__global__ void difference_kernel(float *output, float *vol1, float *vol2, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (nx*ny)*k + i + nx*j;

    if ((i < nx) && (j < ny) && (k < nz)) {
        output[ijk] = (vol1[ijk] - vol2[ijk]) * (vol1[ijk] - vol2[ijk]);
    }
    return;
}

// MAIN HOST FUNCTION //
float cuda_norm(float *input, int nx, int ny, int nz)
{
    int volSize = nx * ny * nz;
    float *d_input;
    float norm;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_input,volSize*sizeof(float));
    cudaMemcpy(d_input,input,volSize*sizeof(float),cudaMemcpyHostToDevice);

    // Measure Norm of Input Volume
    square<float>        unary_op;
    thrust::plus<float>  binary_op;
    thrust::device_vector<float> input_vec(d_input, d_input + volSize);
    norm = std::sqrt( thrust::transform_reduce(input_vec.begin(), input_vec.end(), unary_op, 0.0f, binary_op) );
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaFree(d_input);

    return norm;
}

// MAIN HOST FUNCTION //
float cuda_sum(float *input, int nx, int ny, int nz)
{
    int volSize = nx * ny * nz;
    float *d_input;
    float sum;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_input,volSize*sizeof(float));
    cudaMemcpy(d_input,input,volSize*sizeof(float),cudaMemcpyHostToDevice);

    // Sum up all the Elements
    thrust::plus<float>  binary_op;
    thrust::device_vector<float> input_vec(d_input, d_input + volSize);
    sum = thrust::reduce(input_vec.begin(), input_vec.end(), 0.0f, binary_op);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaFree(d_input);

    return sum;
}

float cuda_rmse(float *recon, float *original, int nx, int ny, int nz)
{
    int volSize = nx * ny * nz;
    float *d_recon, *d_original, *d_diff;
    float rmse;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_recon,volSize*sizeof(float));
    cudaMalloc((void**)&d_original,volSize*sizeof(float));
    cudaMalloc((void**)&d_diff,volSize*sizeof(float));

    cudaMemcpy(d_recon,recon,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_original,original,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_diff, 0.0f, volSize*sizeof(float));

    difference_kernel<<<dimGrid,dimBlock>>>(d_diff, d_recon, d_original, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    // Sum up all the Elements
    thrust::plus<float>  binary_op;
    thrust::device_vector<float> diff_vec(d_diff, d_diff + volSize);
    rmse = thrust::reduce(diff_vec.begin(), diff_vec.end(), 0.0f, binary_op);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaFree(d_recon);
    cudaFree(d_original);
    cudaFree(d_diff);

    return std::sqrt(rmse/(nx*ny*nz));
}

float cuda_euclidean_dist(float *recon, float *original, int nx, int ny, int nz)
{
    int volSize = nx * ny * nz;
    float *d_recon, *d_original, *d_diff;
    float L2;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_recon,volSize*sizeof(float));
    cudaMalloc((void**)&d_original,volSize*sizeof(float));
    cudaMalloc((void**)&d_diff,volSize*sizeof(float));

    cudaMemcpy(d_recon,recon,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_original,original,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_diff, 0.0f, volSize*sizeof(float));

    difference_kernel<<<dimGrid,dimBlock>>>(d_diff, d_recon, d_original, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    // Sum up all the Elements
    thrust::plus<float>  binary_op;
    thrust::device_vector<float> diff_vec(d_diff, d_diff + volSize);
    L2 = thrust::reduce(diff_vec.begin(), diff_vec.end(), 0.0f, binary_op);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaFree(d_recon);
    cudaFree(d_original);
    cudaFree(d_diff);

    cudaDeviceSynchronize();

    return std::sqrt(L2);
}
