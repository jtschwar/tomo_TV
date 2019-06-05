//
//  tlib.hpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef dynamic_tlib_hpp
#define dynamic_tlib_hpp

#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>

void tomography(Eigen::MatrixXf& recon, Eigen::VectorXf& b, Eigen::VectorXf& innerProduct, Eigen::SparseMatrix<float, Eigen::RowMajor>& A, float beta, float max_row);

Eigen::MatrixXf tv2Dderivative(Eigen::MatrixXf recon);

float tv2D(Eigen::MatrixXf& recon);

void circshift(Eigen::MatrixXf input, Eigen::MatrixXf& output, int i, int j);

float rmepsilon(float input);

void parallelRay(int& Nray, Eigen::VectorXf& angles, Eigen::SparseMatrix<float, Eigen::RowMajor>& A);

void removeBadElements(Eigen::VectorXf& xx, Eigen::VectorXf& yy, Eigen::VectorXf I);

void saveResults(Eigen::VectorXf vec, int direc, std::string name);

void saveVec(Eigen::VectorXf vec, std::string name);

float CosAlpha(Eigen::MatrixXf& recon, Eigen::MatrixXf& tv_derivative, Eigen::VectorXf& g, Eigen::VectorXf& b, Eigen::SparseMatrix<float, Eigen::RowMajor>& A);

void poissonNoise(Eigen::VectorXf& sinogram);

void read_parameters(int& Niter,float& Niter_red,int& ng,float& dTheta,float& beta,float& beta_red,float& alpha,float& alpha_red,float& eps,float& r_max);

#endif /* dynamic_tlib_hpp */
