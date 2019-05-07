//
//  tlib.hpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef tlib_hpp
#define tlib_hpp

#include <stdio.h>

void tomography(Eigen::MatrixXf& recon, Eigen::MatrixXf& tiltSeries, Eigen::SparseMatrixBase<float>& A, int beta);

float rmepsilon(float input);

void parallelRay(int& Nray, Eigen::VectorXf& angles);

#endif /* tlib_hpp */
