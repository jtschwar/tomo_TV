//
//  tv_fgb.h
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef tv_fgp
#define tv_fgp

#include <memory.h>

float cuda_tv_fgp_3D(float *vol, int ng, float lambda, int dimX, int dimY, int dimZ, int gpuIndex=-1);

#endif /* tv_fgb_h */
