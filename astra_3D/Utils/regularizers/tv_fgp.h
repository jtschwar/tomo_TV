//
//  astra_ctlib.hpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef tv_fgp
#define tv_fgp

#include <memory.h>

float cuda_tv_fgp(float *recon, int ng, float lambda, int nx, int ny, int nz);

#endif /* tlib_hpp */
