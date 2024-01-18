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
#include <thrust/extrema.h>
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

    int ijk = (ny*nz)*i + nz*j + k;

    if ((i < nx) && (j < ny) && (k < nz)) {
        output[ijk] = (vol1[ijk] - vol2[ijk]) * (vol1[ijk] - vol2[ijk]);
    }
}

__global__ void cuda_positivity_kernel(float *vol, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (ny*nz)*i + nz*j + k;

    if ((i < nx) && (j < ny) && (k < nz)) {
        if (vol[ijk] < 0.0f) { vol[ijk] = 0.0f; }
    }
}

__global__ void cuda_background_kernel(float *vol, int value, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (ny*nz)*i + nz*j + k;

    if ((i < nx) && (j < ny) && (k < nz)) {
        if (vol[ijk] == 0.0f) { vol[ijk] = value; }
    }
}

__global__ void cuda_rescale_volume(float *vol, int value, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (ny*nz)*i + nz*j + k;

    if ((i < nx) && (j < ny) && (k < nz)) {
        vol[ijk] /= value; 
    }
}

__global__ void cuda_soft_thresholding(float *vol, float lambda, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (ny*nz)*i + nz*j + k;
    if ((i < nx) && (j < ny) && (k < nz)) {
        float value = fabs(vol[ijk]); 
        vol[ijk] = signbit(lambda-value)*copysign(value-lambda,vol[ijk]); 
    }
}

__global__ void nesterov_momentum_kernel(float *yk, float *xk, float *xk_old, float beta, int nx, int ny, int nz)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int ijk = (ny*nz)*i + nz*j + k;
    if ((i < nx) && (j < ny) && (k < nz)) {
        yk[ijk] = xk[ijk] + beta * (xk[ijk] - xk_old[ijk]); }
}

// __global__ void cuda_ogm_momentum(float *yk, float *xk, float *xk_old, float beta, float gamma, int nx, int ny, int nz)
// {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     int j = blockDim.y * blockIdx.y + threadIdx.y;
//     int k = blockDim.z * blockIdx.z + threadIdx.z;

//     int ijk = (ny*nz)*i + nz*j + k;
//     if ((i < nx) && (j < ny) && (k < nz)) {
//         yt[ijk] = xt[ijk] + beta * (xt[ijk] - xk_old[ijk]) + gamma * (xt[ijk] - yt[ijk]); }
// }


// MAIN HOST FUNCTION //
float cuda_norm(float *input, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();

        // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

    int volSize = nx * ny * nz;
    float *d_input;
    float norm;

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_input,volSize*sizeof(float));
    cudaMemcpy(d_input,input,volSize*sizeof(float),cudaMemcpyHostToDevice);

    // Measure Norm of Input Volume
    square<float>        unary_op;
    thrust::plus<float>  binary_op;
    thrust::device_vector<float> input_vec(d_input, d_input + volSize);
    norm = thrust::transform_reduce(input_vec.begin(), input_vec.end(), unary_op, 0.0f, binary_op);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaFree(d_input);

    return norm;
}

// MAIN HOST FUNCTION //
float cuda_l1_norm(float *input, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();

        // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

    int volSize = nx * ny * nz;
    float *d_input;
    float norm;

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_input,volSize*sizeof(float));
    cudaMemcpy(d_input,input,volSize*sizeof(float),cudaMemcpyHostToDevice);

    // Measure Norm of Input Volume
    absolute_value<float>   unary_op; 
    thrust::plus<float>  binary_op;
    thrust::device_vector<float> input_vec(d_input, d_input + volSize);
    norm = thrust::transform_reduce(input_vec.begin(), input_vec.end(), unary_op, 0.0f, binary_op);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaFree(d_input);

    return norm;
}


// MAIN HOST FUNCTION //
float cuda_sum(float *input, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();

        // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

    int volSize = nx * ny * nz;
    float *d_input;
    float sum;

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

float cuda_rmse(float *recon, float *original, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();

        // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

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

    return rmse;

    // return std::sqrt(rmse/(nx*ny*nz));
}

float cuda_euclidean_dist(float *vol1, float *vol2, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();

        // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

    int volSize = nx * ny * nz;
    float *d_vol1, *d_vol2, *d_diff;
    float L2;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_vol1,volSize*sizeof(float));
    cudaMalloc((void**)&d_vol2,volSize*sizeof(float));
    cudaMalloc((void**)&d_diff,volSize*sizeof(float));

    cudaMemcpy(d_vol1,vol1,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vol2,vol2,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemset(d_diff, 0.0f, volSize*sizeof(float));

    difference_kernel<<<dimGrid,dimBlock>>>(d_diff, d_vol1, d_vol2, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    // Sum up all the Elements
    thrust::plus<float>  binary_op;
    thrust::device_vector<float> diff_vec(d_diff, d_diff + volSize);
    L2 = thrust::reduce(diff_vec.begin(), diff_vec.end(), 0.0f, binary_op);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaFree(d_vol1);
    cudaFree(d_vol2);
    cudaFree(d_diff);

    cudaDeviceSynchronize();

    return L2;
}

void cuda_positivity(float *recon, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();
    }

    int volSize = nx * ny * nz;
    float *d_recon;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_recon,volSize*sizeof(float));
    cudaMemcpy(d_recon, recon, volSize*sizeof(float), cudaMemcpyHostToDevice);

    cuda_positivity_kernel<<<dimGrid,dimBlock>>>(d_recon, nx, ny, nz);
    cudaDeviceSynchronize();
    cudaPeekAtLastError();

    cudaMemcpy(recon, d_recon, volSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_recon);
    cudaDeviceSynchronize();
}


void cuda_set_background(float *vol, int value, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();
    }

    int volSize = nx * ny * nz;
    float *d_vol;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    // allocate space for volume on device
    cudaMalloc((void**)&d_vol,volSize*sizeof(float));
    cudaMemcpy(d_vol, vol, volSize*sizeof(float), cudaMemcpyHostToDevice);

    cuda_background_kernel<<<dimGrid,dimBlock>>>(d_vol, value, nx, ny, nz);
    cudaDeviceSynchronize(); cudaPeekAtLastError();

    cudaMemcpy(vol, d_vol, volSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vol);
    cudaDeviceSynchronize();
}

void cuda_rescale_volume(float *vol, float *original_vol, int nx, int ny, int nz, int ne, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();
    }

    float maxValVol, maxValOVol;
    int volSize = nx * ny * nz;
    float *d_vol;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    cudaMalloc((void**)&d_vol,volSize*sizeof(float));

    // Rescale all Elements in Volume by Ground Truth
    for (int e =0; e < ne; e++) {
        cudaMemcpy(d_vol, &original_vol[volSize*e], volSize*sizeof(float), cudaMemcpyHostToDevice);
        thrust::device_vector<float> input_Ovec(d_vol, d_vol + volSize);
        maxValOVol = *thrust::max_element(input_Ovec.begin(), input_Ovec.end());

        cudaMemcpy(d_vol, &vol[volSize*e], volSize*sizeof(float), cudaMemcpyHostToDevice);
        thrust::device_vector<float> input_vec(d_vol, d_vol + volSize);
        maxValVol = *thrust::max_element(input_vec.begin(), input_vec.end());

        cuda_rescale_volume<<<dimGrid,dimBlock>>>(d_vol, maxValVol/maxValOVol, nx, ny, nz);        
        cudaDeviceSynchronize(); cudaPeekAtLastError();

        cudaMemcpy(&vol[volSize*e], d_vol, volSize*sizeof(float), cudaMemcpyDeviceToHost);
    }
}

void cuda_soft_threshold(float *vol, float lambda, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();
    }

    int volSize = nx * ny * nz;
    float *d_vol;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    // allocate space for volume on device
    cudaMalloc((void**)&d_vol,volSize*sizeof(float));
    cudaMemcpy(d_vol, vol, volSize*sizeof(float), cudaMemcpyHostToDevice);

    cuda_soft_thresholding<<<dimGrid,dimBlock>>>(d_vol, lambda, nx, ny, nz);
    cudaDeviceSynchronize(); cudaPeekAtLastError();

    cudaMemcpy(vol, d_vol, volSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vol);
    cudaDeviceSynchronize();
}

void cuda_nesterov_momentum(float *yt, float *xt, float *xt_old, float beta, int nx, int ny, int nz, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();
    }

    int volSize = nx * ny * nz;
    float *d_yt, *d_xt, *d_xt_old;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    // allocate space for volume on device
    cudaMalloc((void**)&d_yt,volSize*sizeof(float));
    cudaMalloc((void**)&d_xt,volSize*sizeof(float));
    cudaMalloc((void**)&d_xt_old,volSize*sizeof(float));

    cudaMemcpy(d_yt,yt,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_xt,xt,volSize*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_xt_old,xt_old,volSize*sizeof(float),cudaMemcpyHostToDevice);

    nesterov_momentum_kernel<<<dimGrid,dimBlock>>>(d_yt, d_xt, d_xt_old, beta, nx, ny, nz);
    cudaDeviceSynchronize(); cudaPeekAtLastError();

    cudaMemcpy(yt, d_yt, volSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xt, d_xt, volSize*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(xt_old, d_xt_old, volSize*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_yt); cudaFree(d_xt); cudaFree(d_xt_old);
    cudaDeviceSynchronize();
}

// 4D Functions
//////////////////////////////////////////////////////////////////////////////////////
void cuda_positivity_4D(float *recon, int nx, int ny, int nz, int ne, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();
    }

    int volSize = nx * ny * nz;
    float *d_recon;

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_recon,volSize*sizeof(float));

    // Iterate through all elements
    for (int e=0; e<ne; e++) {
        cudaMemcpy(d_recon, &recon[volSize*e], volSize*sizeof(float), cudaMemcpyHostToDevice);

        cuda_positivity_kernel<<<dimGrid,dimBlock>>>(d_recon, nx, ny, nz);
        cudaDeviceSynchronize();
        cudaPeekAtLastError();

        cudaMemcpy(&recon[volSize*e], d_recon, volSize*sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_recon);
    cudaDeviceSynchronize();
}

float *cuda_rmse_4D(float *recon, float *original, int nx, int ny, int nz, int ne, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
    }

    int volSize = nx * ny * nz;
    float *d_recon, *d_original, *d_diff;
    
    // Initialize Array for Measuring RMSE
    float *rmse = new float[ne];

    // Block
    dim3 dimBlock(BLKXSIZE,BLKXSIZE, BLKXSIZE);

    // Grid
    dim3 dimGrid(idivup(nx,BLKXSIZE), idivup(ny,BLKXSIZE), idivup(nz,BLKXSIZE));

    /*allocate space for volume on device*/
    cudaMalloc((void**)&d_recon,volSize*sizeof(float));
    cudaMalloc((void**)&d_original,volSize*sizeof(float));
    cudaMalloc((void**)&d_diff,volSize*sizeof(float));

    // Create Summation Operator
    thrust::plus<float>  binary_op;

    // Iterate through all elements
    for (int e=0; e<ne; e++){

        cudaMemcpy(d_recon, &recon[volSize*e], volSize*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_original,&original[volSize*e],volSize*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemset(d_diff, 0.0f, volSize*sizeof(float));


        // Calculate Element-Wise Difference
        difference_kernel<<<dimGrid,dimBlock>>>(d_diff, d_recon, d_original, nx, ny, nz);
        cudaDeviceSynchronize();
        cudaPeekAtLastError();

        // Sum up all the Elements
        thrust::device_vector<float> diff_vec(d_diff, d_diff + volSize);
        rmse[e] = std::sqrt(thrust::reduce(diff_vec.begin(), diff_vec.end(), 0.0f, binary_op)/volSize);
        
        cudaDeviceSynchronize();
        cudaPeekAtLastError();
    }

    cudaFree(d_recon);
    cudaFree(d_original);
    cudaFree(d_diff);
    cudaDeviceSynchronize();

    return rmse;
}