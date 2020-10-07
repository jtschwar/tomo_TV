//
//  tv_fgb.h
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef tv_fgp
#define tv_fgp

#include <memory.h>

void cuda_tv_fgp(float *vol, float lambda, int ng, int methodTV, int nonneg, int dimX, int dimY, int dimZ, int gpuIndex);

#endif /* tv_fgb_h */
