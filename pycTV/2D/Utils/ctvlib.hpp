
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
	ctvlib(int Nrow, int Ncol);

	// Constructs Measurement Matrix. 
	SpMat parallelRay(int Nray, Eigen::VectorXf angles);
    void loadA(Eigen::Ref<Mat> pyA);
	void normalization();

	// 2D Reconstruction Algorithms
	void ART(Eigen::Ref<Eigen::VectorXf> recon, Eigen::Ref<Eigen::VectorXf> b, double beta);
    void ART2(Eigen::Ref<Eigen::VectorXf> recon, Eigen::Ref<Eigen::VectorXf> b, double beta, int max_row);
    void SIRT(Eigen::Ref<Eigen::VectorXf> recon, Eigen::Ref<Eigen::VectorXf> b, double beta);
    Eigen::VectorXf forwardProjection(Eigen::Ref<Eigen::VectorXf> recon, int max_row);

	// Functions For Constructing Measurement Matrix. 
	float rmepsilon(float input);
	void removeBadElements(Eigen::VectorXf& xx, Eigen::VectorXf& yy, Eigen::VectorXf I);

    // Functions for TV
    Eigen::MatrixXf tv_loop(Eigen::MatrixXf&, float dPOCS, int ng);
    Eigen::MatrixXf tv2Dderivative(Eigen::MatrixXf recon);
    void circshift(Eigen::MatrixXf input, Eigen::MatrixXf& output, int i, int j);
    
    // Analysis Algorithms
    float CosAlpha(Eigen::MatrixXf& recon,  Eigen::VectorXf& b, Eigen::VectorXf& g, int max_row);
    
	// Member Variables. 
	SpMat A;
	int Nrow, Ncol;
	Eigen::VectorXf innerProduct;
    
};

#endif /* tlib_hpp */
