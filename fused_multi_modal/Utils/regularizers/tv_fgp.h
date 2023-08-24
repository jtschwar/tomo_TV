//
//  tv_fgb.h
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef tv_fgp
#define tv_fgp

#include <memory.h>

float cuda_tv_fgp_4D(float *vol, int iter, float lambdaPar, int dimX, int dimY, int dimZ, int Nel, int gpuIndex=-1);

#endif /* tv_fgb_h */
