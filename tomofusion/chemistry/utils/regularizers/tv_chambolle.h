//
//  tv_chambolle.h
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef tv_chambolle
#define tv_chambolle

#include <memory.h>

float cuda_tv_gd_chambolle(float *recon, int ng, float lambda, int nx, int ny, int nz);

#endif /* tlib_hpp */
