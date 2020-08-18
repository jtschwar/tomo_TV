//
//  matrix_ops.h
//
//  Created by Hovden Group on 8/17/2020.
//  Copyright © 2020 Jonathan Schwartz. All rights reserved.
//

#ifndef matrix_ops
#define matrix_ops

#include <memory.h>

float cuda_norm(float *input, int nx, int ny, int nz);

float cuda_sum(float *input, int nx, int ny, int nz);

float cuda_euclidean_dist(float *recon, float *original, int nx, int ny, int nz);

float cuda_rmse(float *recon, float *original, int nx, int ny, int nz);

#endif /* tlib_hpp */
