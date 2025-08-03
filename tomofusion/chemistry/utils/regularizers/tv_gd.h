//
//  astra_ctlib.hpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef tv_gd
#define tv_gd

#include <memory.h>

float cuda_tv_3D(float *recon, int nx, int ny, int nz, int gpuIndex=-1);

float cuda_tv_gd_4D(float *recon, int ng, float dPOCS, int nx, int ny, int nz, int Nel, int gpuIndex=-1);

#endif /* tlib_hpp */
