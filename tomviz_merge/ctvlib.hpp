
//
//  ctvlib.hpp
//
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

    // Member Variables.
    Eigen::VectorXf *recon, *temp_recon, *tv_recon;
    SpMat A;
    int Nrow, Ncol, Nslice, Ny, Nz;
    Eigen::VectorXf innerProduct;
    Mat b, g;
    SpMat M; // Diagonal Weight Matrix for SIRT;
    
	// Constructor
	ctvlib(int Nslice, int Nray, int Nproj);
    
    int get_Nslice();
    int get_Nray();
    
    // Initialize Additional Volumes
    void initialize_recon_copy();
    void initialize_tv_recon();

	// Initialize Experimental Projections. 
	void set_tilt_series(Mat in);

	// Constructs Measurement Matrix.
    void loadA(Eigen::Ref<Mat> pyA);
    void update_proj_angles(Eigen::Ref<Mat> pyA, int Nproj);
	void normalization();
    float lipschits();

	// 2D Reconstructions
	void ART(float beta);
    void randART(float beta);
    void SIRT(float beta);
    void positivity();
    
    // Stochastic Reconstruction
    std::vector<int> calc_proj_order(int n);
    
	//Forward Project Reconstruction for Data Tolerance Parameter. 
	void forward_projection();

    // Acquire local copy of reconstruction.
    void copy_recon();
    
    // Measure 2-norm of projections and reconstruction.
    float matrix_2norm();
    float data_distance();
    
    // Total variation
    void tv_gd_3D(int ng, float dPOCS);
    
    // Return reconstruction to python.
    Mat get_recon(int i);
    
    // Return projections to python. 
    Mat get_projections();
    
    // Set Slices to Zero.
    void restart_recon();
    
};

#endif /* ctvlib_hpp */
