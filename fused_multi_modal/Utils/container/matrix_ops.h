//
//  matrix_ops.h
//
//  Created by Hovden Group on 8/17/2020.
//  Copyright © 2020 Jonathan Schwartz. All rights reserved.
//

#ifndef matrix_ops
#define matrix_ops

#include <memory.h>

float cuda_norm(float *input, int nx, int ny, int nz, int gpuIndex=-1);

float cuda_l1_norm(float *input, int nx, int ny, int nz, int gpuIndex=-1);

float cuda_sum(float *input, int nx, int ny, int nz, int gpuIndex=-1);

float cuda_euclidean_dist(float *recon, float *original, int nx, int ny, int nz, int gpuIndex=-1);

float cuda_rmse(float *recon, float *original, int nx, int ny, int nz, int gpuIndex=-1);

void cuda_positivity(float *recon, int nx, int ny, int nz, int gpuIndex=-1);

void cuda_set_background(float *vol, int value, int nx, int ny, int nz, int gpuIndex=-1);

void cuda_rescale_volume(float *vol, float *original_vol, int nx, int ny, int nz, int ne, int gpuIndex=-1);

void cuda_soft_threshold(float *vol, float lambda, int nx, int ny, int nz, int gpuIndex=-1);

void cuda_nesterov_momentum(float *yt, float *xt, float *xt_old, float beta, int nx, int ny, int nz, int gpuIndex=-1);

//////////////////////////////////////////////////////////////////////

void cuda_positivity_4D(float *recon, int nx, int ny, int nz, int ne=1, int gpuIndex=-1);

float *cuda_rmse_4D(float *recon, float *original, int nx, int ny, int nz, int ne=1, int gpuIndex=-1);

#endif /* tlib_hpp */
