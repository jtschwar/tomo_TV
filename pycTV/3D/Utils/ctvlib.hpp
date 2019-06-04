
//
//  tlib.hpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef ctvlib_hpp
#define ctvlib_hpp

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

class ctvlib 
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat;

public: 

	// Initializes Measurement Matrix. 
	ctvlib(int Nx, int Ny, int Nz);

	// Initialize Experimental Projections. 
	void setTiltSeries(Mat in);

	// Constructs Measurement Matrix. 
	void parallelRay(int Nray, Eigen::VectorXf angles);
	void normalization();

	// 2D ART Reconstruction 
	Mat recon(Eigen::Ref<Eigen::VectorXf> recon, double beta, int s, int Nx);

	// Functions For Constructing Measurement Matrix. 
	float rmepsilon(float input);
	void removeBadElements(Eigen::VectorXf& xx, Eigen::VectorXf& yy, Eigen::VectorXf I);

	//Forward Project Reconstruction for Data Tolerance Parameter. 
	Eigen::VectorXf forwardProjection(Eigen::Ref<Eigen::VectorXf> recon);

	// Member Variables. 
	SpMat A;
	int Nrow, Ncol;
	Eigen::VectorXf innerProduct;
	Mat b;
    
};

#endif /* tlib_hpp */
