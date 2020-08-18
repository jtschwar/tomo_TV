//
//  astra_ctlib.hpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef tv_gd
#define tv_gd

#include <memory.h>

float cuda_tv_gd_3D(float *recon, int ng, float dPOCS, int nx, int ny, int nz);

float cuda_tv_3D(float *recon, int nx, int ny, int nz);

#endif /* tlib_hpp */
