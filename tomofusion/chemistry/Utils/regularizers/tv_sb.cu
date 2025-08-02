/* CUDA implementation of Split Bregman - TV denoising-regularisation model (2D/3D) [1]
*
* Input Parameters:
* 1. Noisy image/volume
* 2. lambda - regularisation parameter
* 3. Number of iterations [OPTIONAL parameter]
* 4. eplsilon - tolerance constant [OPTIONAL parameter]
* 5. TV-type: 'iso' or 'l1' [OPTIONAL parameter]
* 7. GPU device number if for multigpu run (default 0)
* Output:
* [1] Filtered/regularized image/volume
*
* [1]. Goldstein, T. and Osher, S., 2009. The split Bregman method for L1-regularized problems. SIAM journal on imaging sciences, 2(2), pp.323-343.
*/
// 
// tv_sb.cu
//
// Created by Hovden Group on 2/17/23
// Adapted fromn CCPI-Regularization-Toolkit (https://github.com/vais-ral/CCPi-Regularisation-Toolkit)
//

#include "tv_sb.h"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#define BLKSIZE 8

#define idivup(a, b) ( ((a)%(b) != 0) ? (a)/(b)+1 : (a)/(b) )

__global__ void gauss_seidel3D_kernel(float *U, float *A, float *U_prev, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, float lambda, float mu, float normConst, int N, int M, int Z, int ImSize)
{

    // Gauss Siedel Solution: 
    // G_{i,j} = lambda / (mu + 4lambda) * (sum_u + sum_d + sum_b) + mu / (mu + 4lambda) * f_{i,j,k} 

    float sum, d_val, b_val;
    int i1, i2, j1, j2, k1, k2;

    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 
    if ((i < N) && (j < M) && (k < Z)) {

        i1 = i+1; if (i1 == N) i1 = i-1;    i2 = i-1; if (i2 < 0) i2 = i+1;
        j1 = j+1; if (j1 == M) j1 = j-1;    j2 = j-1; if (j2 < 0) j2 = j+1;
        k1 = k+1; if (k1 == Z) k1 = k-1;    k2 = k-1; if (k2 < 0) k2 = k+1;

        // d_val = Dx[(N*M)*k + j*N+i2] - Dx[index] + Dy[(N*M)*k + j2*N+i] - Dy[index] + Dz[(N*M)*k2 + j*N+i] - Dz[index];
        // b_val = -Bx[(N*M)*k + j*N+i2] + Bx[index] - By[(N*M)*k + j2*N+i] + By[index] - Bz[(N*M)*k2 + j*N+i] + Bz[index];

        // sum_d = dx_{i-1,j,k} - dx_{i-1,j,k} + dy_{i,j-1,k} - by_{i,j,k} + bz_{i,j,k-1} - bz_{i,j,k}
        d_val = Dx[(Z*M)*i + Z*j + i2] - Dx[index] + Dy[(Z*M)*k + Z*j2 + i] - Dy[index] + Dz[(Z*M)*i + Z*j + k2] - Dz[index];
        
        // sum_b = bx_{i,j,k} - bx_{i-1,j,k} + by_{i,j,k} - by_{i,j-1,k} + bz_{i,j,k} - bz_{i,j,k} 
        b_val = -Bx[(Z*M)*i2 + Z*j + k] + Bx[index] - By[(Z*M)*i + Z*j2 + i] + By[index] - Bz[(N*M)*i + Z*j + k2] + Bz[index];

        sum = d_val + b_val;
        
        // sum_u = u_{i+1,j,k} + u_{i-1,j,k} + u_{i,j+1,k} + u_{i,j-1,k} + u_{i,j,k+1} + u_{i,j,k-1}
        sum += U_prev[(Z*M)*k + Z*j + i1] + U_prev[(Z*M)*k + Z*j + i2] + U_prev[(Z*M)*k + Z*j1 + i] + U_prev[(Z*M)*k + Z*j2 + i] + U_prev[(Z*M)*k1 + Z*j + i] + U_prev[(N*M)*k2 + j*N+i];
        
        sum *= lambda;
        sum += mu*A[index];
        
        U[index] = normConst*sum;
    }
}

__global__ void updDxDy_shrinkIso3D_kernel(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, float lambda, int N, int M, int Z, int ImSize)
{

    int i1,j1,k1;
    float val1, val11, val2, val3, denom_lam, denom;
    denom_lam = 1.0f/lambda;

    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if ((i < N) && (j < M) && (k < Z)) {
        i1 = i+1; if (i1 == N) i1 = i-1;
        j1 = j+1; if (j1 == M) j1 = j-1;
        k1 = k+1; if (k1 == Z) k1 = k-1;

            // \nabla_i u^k + b_i
            val1 = (U[(Z*M)*i1 + Z*j + k] - U[index]) + Bx[index];
            val2 = (U[(Z*M)*i + Z*j1 + k] - U[index]) + By[index];
            val3 = (U[(Z*M)*i + Z*j + k1] - U[index]) + Bz[index];
            
            // s^k = sqrt(\sum_{i in (x,y,z)} (\nabla_i u)^2)
            denom = sqrt(val1*val1 + val2*val2 + val3*val3);
            
            // max(s^k - 1/lambda)
            val11 = (denom - denom_lam); if (val11 < 0.0f) val11 = 0.0f;
            
            if (denom != 0.0f) {
                Dx[index] = val11*(val1/denom);
                Dy[index] = val11*(val2/denom);
                Dz[index] = val11*(val3/denom);
            }
            else {
                Dx[index] = 0.0f;   Dy[index] = 0.0f;
                Dz[index] = 0.0f;
            }
    }
}

__global__ void updBxBy3D_kernel(float *U, float *Dx, float *Dy, float *Dz, float *Bx, float *By, float *Bz, int N, int M, int Z, int ImSize)
{
    int i1,j1,k1;

    //calculate each thread global index
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if ((i < N) && (j < M) && (k < Z)) {
            /* symmetric boundary conditions (Neuman) */
            i1 = i+1; if (i1 == N) i1 = i-1;
            j1 = j+1; if (j1 == M) j1 = j-1;
            k1 = k+1; if (k1 == Z) k1 = k-1;

            // b_{x}^{k+1} = b_{x}^{k} + (\nabla_x u^{k+1} - d_x^{k+1}
            Bx[index] += (U[(Z*M)*i1 + N*j + k ] - U[index]) - Dx[index];
            // b_{y}^{k+1} = b_{y}^{k} + (\nabla_y u^{k+1} - d_y^{k+1}
            By[index] += (U[(Z*M)*i + Z*j1 + i ] - U[index]) - Dy[index];
            // b_{z}^{k+1} = b_{z}^{k} + (\nabla_z u^{k+1} - d_z^{k+1}
            Bz[index] += (U[(Z*M)*i + Z*j + k1 ] - U[index]) - Dz[index];
    }
}

__global__ void SBcopy_kernel3D(float *Input, float* Output, int N, int M, int Z, int num_total)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if (index < num_total)	{
        Output[index] = Input[index];
    }
}

__global__ void SBResidCalc3D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int Z, int num_total)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if (index < num_total)	{
        Output[index] = Input1[index] - Input2[index];
    }
}

float cuda_tv_sb_3D(float *vol, int iter, float mu, float epsil, int dimX, int dimY, int dimZ, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();
    
        // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

	float re, lambda, normConst;

    re = 0.0f;
    int count = 0;
    mu = 1.0f/mu;
    lambda = 2.0f*mu;
    int volSize = dimX*dimY*dimZ;
    float *d_input, *d_update;
    float *Dx=NULL, *Dy=NULL, *Dz=NULL, *Bx=NULL, *By=NULL, *Bz=NULL;

    dim3 dimBlock(BLKSIZE,BLKSIZE,BLKSIZE);
    dim3 dimGrid(idivup(dimX,BLKXSIZE), idivup(dimY,BLKXSIZE),idivup(dimZ,BLKXSIZE));

    /*allocate space for images on device*/
    checkCudaErrors( cudaMalloc((void**)&d_input,volSize*sizeof(float)) );  checkCudaErrors( cudaMalloc((void**)&d_update,volSize*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&Dx,volSize*sizeof(float)) );       checkCudaErrors( cudaMalloc((void**)&Bx,volSize*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&Dy,volSize*sizeof(float)) );       checkCudaErrors( cudaMalloc((void**)&By,volSize*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&Dz,volSize*sizeof(float)) );       checkCudaErrors( cudaMalloc((void**)&Bz,volSize*sizeof(float)) );

    // Initialize Values
    checkCudaErrors( cudaMemcpy(d_input,Input,volSize*sizeof(float),cudaMemcpyHostToDevice));   checkCudaErrors( cudaMemcpy(d_update,Input,volSize*sizeof(float),cudaMemcpyHostToDevice));
    cudaMemset(Dx, 0, volSize*sizeof(float));  cudaMemset(Dy, 0, volSize*sizeof(float));  cudaMemset(Dz, 0, volSize*sizeof(float));
    cudaMemset(Bx, 0, volSize*sizeof(float));  cudaMemset(By, 0, volSize*sizeof(float));  cudaMemset(Bz, 0, volSize*sizeof(float));

    /* The main kernel */
    normConst = 1.0f/(mu + 6.0f*lambda);
    for (int ll = 0; ll < iter; ll++) {

        /* storing old value */
        SBcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, dimX, dimY, dimZ, volSize);
        checkCudaErrors( cudaDeviceSynchronize() );     checkCudaErrors(cudaPeekAtLastError() );

		 /* perform two GS iterations (normally 2 is enough for the convergence) */
        gauss_seidel3D_kernel<<<dimGrid,dimBlock>>>(d_update, d_input, d_update_prev, Dx, Dy, Dz, Bx, By, Bz, lambda, mu, normConst, dimX, dimY, dimZ, volSize);
        checkCudaErrors( cudaDeviceSynchronize() );     checkCudaErrors(cudaPeekAtLastError() );

        /* storing old value */
        SBcopy_kernel3D<<<dimGrid,dimBlock>>>(d_update, d_update_prev, dimX, dimY, dimZ, volSize);
        checkCudaErrors( cudaDeviceSynchronize() );     checkCudaErrors(cudaPeekAtLastError() );
        
        /* 2nd GS iteration */
        gauss_seidel3D_kernel<<<dimGrid,dimBlock>>>(d_update, d_input, d_update_prev, Dx, Dy, Dz, Bx, By, Bz, lambda, mu, normConst, dimX, dimY, dimZ, volSize);
        checkCudaErrors( cudaDeviceSynchronize() );     checkCudaErrors(cudaPeekAtLastError() );

        /* TV-related step */
        updDxDy_shrinkIso3D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Dz, Bx, By, Bz, lambda, dimX, dimY, dimZ, volSize);
        // if (methodTV == 1)  updDxDy_shrinkAniso3D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Dz, Bx, By, Bz, lambda, dimX, dimY, dimZ, DimTotal);
        // else updDxDy_shrinkIso3D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Dz, Bx, By, Bz, lambda, dimX, dimY, dimZ, DimTotal);

        /* update for Bregman variables */
        updBxBy3D_kernel<<<dimGrid,dimBlock>>>(d_update, Dx, Dy, Dz, Bx, By, Bz, dimX, dimY, dimZ, volSize);
        checkCudaErrors( cudaDeviceSynchronize() );     checkCudaErrors(cudaPeekAtLastError() );
    }

    //copy result matrix from device to host memory
    cudaMemcpy(vol,d_update,volSize*sizeof(float),cudaMemcpyDeviceToHost);

    cudaFree(d_input); cudaFree(d_update);
    cudaFree(Dx);   cudaFree(Dy);   cudaFree(Dz);
    cudaFree(Bx);   cudaFree(By);   cudaFree(Bz);

    cudaDeviceSynchronize();

}      
