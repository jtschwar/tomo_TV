/* CUDA implementation of FGP-TV [1] denoising/regularization model (3D case)
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
//  tv_fgp.cu
//
//  Created by Hovden Group on 9/1/20.
//  Adapted from CCPI-Regularization-Toolkit
//


#include "tv_fgp.h"
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


__global__ void Obj_func3D_kernel(float *Ad, float *D, float *R1, float *R2, float *R3, int N, int M, int Z, int ImSize, float lambda)
{

    float val1, val2, val3;

    //calculate each thread global index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if ((i < N) && (j < M) && (k < Z)) {
        if (i <= 0) {val1 = 0.0f;} else {val1 = R1[(Z*M)*(i-1) + Z*j + k];}
        if (j <= 0) {val2 = 0.0f;} else {val2 = R2[(Z*M)*i + Z*(j-1) + k];}
        if (k <= 0) {val3 = 0.0f;} else {val3 = R3[(Z*M)*i + Z*j + (k-1)];}

        //Write final result to global memory
        D[index] = Ad[index] - lambda*(R1[index] + R2[index] + R3[index] - val1 - val2 - val3);
    }
    return;
}

__global__ void Grad_func3D_kernel(float *P1, float *P2, float *P3, float *D, int N, int M, int Z, int ImSize, float multip)
{

    float val1,val2,val3;

    //calculate each thread global index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if ((i < N) && (j < M) && (k <  Z)) {
        // boundary conditions 
        if (i >= N-1) val1 = 0.0f; else val1 = D[index] - D[(Z*M)*(i+1) + Z*j + k];
        if (j >= M-1) val2 = 0.0f; else val2 = D[index] - D[(Z*M)*i + Z*(j+1) + k];
        if (k >= Z-1) val3 = 0.0f; else val3 = D[index] - D[(Z*M)*i + Z*j + (k+1)];

        //Write final result to global memory
        P1[index] += multip * val1;
        P2[index] += multip * val2;
        P3[index] += multip * val3;
    }
    return;
}

__global__ void Proj_func3D_iso_kernel(float *P1, float *P2, float *P3, int N, int M, int Z, int ImSize)
{

    float denom,sq_denom;
    //calculate each thread global index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if ((i < N) && (j < M) && (k <  Z)) {
        denom = pow(P1[index],2) +  pow(P2[index],2) + pow(P3[index],2);

        if (denom > 1.0f) {
            sq_denom = 1.0f/sqrt(denom);
            P1[index] = P1[index]*sq_denom;
            P2[index] = P2[index]*sq_denom;
            P3[index] = P3[index]*sq_denom;
        }
    }
    return;
}

__global__ void Proj_func3D_aniso_kernel(float *P1, float *P2, float *P3, int N, int M, int Z, int ImSize)
{

    float val1, val2, val3;
    //calculate each thread global index
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if ((i < N) && (j < M) && (k <  Z)) {
                val1 = abs(P1[index]);
                val2 = abs(P2[index]);
                val3 = abs(P3[index]);
                if (val1 < 1.0f) {val1 = 1.0f;}
                if (val2 < 1.0f) {val2 = 1.0f;}
                if (val3 < 1.0f) {val3 = 1.0f;}
                P1[index] = P1[index]/val1;
                P2[index] = P2[index]/val2;
                P3[index] = P3[index]/val3;
    }
    return;
}


__global__ void nonneg3D_kernel(float* Output, int N, int M, int Z, int num_total)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if (index < num_total)  {
        if (Output[index] < 0.0f) Output[index] = 0.0f;
    }
}

__global__ void FGPResidCalc3D_kernel(float *Input1, float *Input2, float* Output, int N, int M, int Z, int num_total)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;

    int index = (Z*M)*i + Z*j + k; 

    if (index < num_total)  {
        Output[index] = Input1[index] - Input2[index];
    }
}

//Measure Reconstruction's TV.
__global__ void tv_kernel_3D(float *vol, float *tv_recon, int nx, int ny, int nz)
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
}

////////////MAIN HOST FUNCTION ///////////////
float cuda_tv_fgp_4D(float *vol, int iter, float lambdaPar, int dimX, int dimY, int dimZ, int Nel, int gpuIndex)
{
    // Set GPU Index
    if (gpuIndex != -1) {
        cudaSetDevice(gpuIndex);
        cudaError_t err = cudaGetLastError();
    
	    // Ignore errors caused by calling cudaSetDevice multiple times
        if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess)
            return false;
    }

    int nonneg = 1, methodTV = 0;
    float multip, tv = 0;

    /*3D verson*/
    int ImSize = dimX*dimY*dimZ;
    float *d_input, *d_update=NULL, *P1=NULL, *P2=NULL, *P3=NULL;

    // Look into, originally BLK(X/Y/Z)SIZE
    dim3 dimBlock(BLKXSIZE,BLKXSIZE,BLKXSIZE);
    dim3 dimGrid(idivup(dimX,BLKXSIZE), idivup(dimY,BLKXSIZE),idivup(dimZ,BLKXSIZE));

    /*allocate space for images on device*/
    cudaMalloc((void**)&d_input,ImSize*sizeof(float));
    cudaMalloc((void**)&d_update,ImSize*sizeof(float));
    cudaMalloc((void**)&P1,ImSize*sizeof(float));
    cudaMalloc((void**)&P2,ImSize*sizeof(float));
    cudaMalloc((void**)&P3,ImSize*sizeof(float));

    // // Operators for Global Reductions
    thrust::plus<float>  binary_op;

    multip = (1.0f/(26.0f*lambdaPar));

    for (int e=0; e<Nel; e++) 
    {

        cudaMemcpy(d_input,&vol[e*ImSize],ImSize*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemset(d_update, 0, ImSize*sizeof(float));
        cudaMemset(P1, 0, ImSize*sizeof(float));
        cudaMemset(P2, 0, ImSize*sizeof(float));
        cudaMemset(P3, 0, ImSize*sizeof(float));

        // Measure TV (in this case d_update == tv_recon)
        tv_kernel_3D<<<dimGrid,dimBlock>>>(d_input, d_update, dimX, dimY, dimZ);
        cudaDeviceSynchronize();
        cudaPeekAtLastError();

        // Measure Norm of TV - Gradient
        thrust::device_vector<float> tv_vec(d_update, d_update + ImSize);
        tv += thrust::reduce(tv_vec.begin(), tv_vec.end(), 0.0f, binary_op);

        /********************** Run CUDA 3D kernel here ********************/

        /* Main Loop */
        for (int i = 0; i < iter; i++) {

            /* computing the gradient of the objective function */
            Obj_func3D_kernel<<<dimGrid,dimBlock>>>(d_input, d_update, P1, P2, P3, dimX, dimY, dimZ, ImSize, lambdaPar);
            cudaDeviceSynchronize();
            cudaPeekAtLastError();

            // Apply Nonnegativity 
            if (nonneg != 0) {
                nonneg3D_kernel<<<dimGrid,dimBlock>>>(d_update, dimX, dimY, dimZ, ImSize);
                cudaDeviceSynchronize();
                cudaPeekAtLastError(); }

            /*Taking a step towards minus of the gradient*/
            Grad_func3D_kernel<<<dimGrid,dimBlock>>>(P1, P2, P3, d_update, dimX, dimY, dimZ, ImSize, multip);
            cudaDeviceSynchronize();
            cudaPeekAtLastError();

            /* projection step */
            if (methodTV == 0) Proj_func3D_iso_kernel<<<dimGrid,dimBlock>>>(P1, P2, P3, dimX, dimY, dimZ, ImSize); /* isotropic kernel */
            else Proj_func3D_aniso_kernel<<<dimGrid,dimBlock>>>(P1, P2, P3, dimX, dimY, dimZ, ImSize); /* anisotropic kernel */
            cudaDeviceSynchronize();
            cudaPeekAtLastError();

        }

        //copy result matrix from device to host memory
        cudaMemcpy(&vol[e*ImSize],d_update,ImSize*sizeof(float),cudaMemcpyDeviceToHost);
    }

    /***************************************************************/

    cudaFree(d_input);
    cudaFree(d_update);
    cudaFree(P1);
    cudaFree(P2);
    cudaFree(P3);

    return tv;
}
