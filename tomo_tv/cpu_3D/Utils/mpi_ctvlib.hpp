
//
//  mpi_ctvlib.hpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef mpi_ctvlib_hpp
#define mpi_ctvlib_hpp

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

class mpi_ctvlib
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat;
    
public:
    
    // Member Variables.
    Mat *recon, *temp_recon, *tv_recon, *original_volume;
    Eigen::VectorXf innerProduct;
    SpMat A, M;
    Mat b, g;
    
    unsigned int Nrow, Ncol, Nslice, Nslice_loc, Ny, Nz, first_slice, last_slice;
    int nproc, rank, size;
    
    // Constructor and MPI Local Properties.
	mpi_ctvlib(int Nslice, int Nray, int Nproj);
    int get_Nslice_loc();
    int get_first_slice();
    int get_rank();
    int get_nproc();
    
    // Initialize Additional Volumes
    void initialize_recon_copy();
    void initialize_original_volume();
    void initialize_tv_recon();
	
    // Initialize Experimental Projections.
	void set_tilt_series(Mat in);
    void set_original_volume(Mat in, int slice);
    void create_projections();
    void poisson_noise(int Nc);

	// Constructs Measurement Matrix.
    void loadA(Eigen::Ref<Mat> pyA);
	void normalization();
    float lipschits();
    void cimminos_method();

    // Dynamic Tomography, update Measurement Matrix and Weights.
    void update_proj_angles(Eigen::Ref<Mat> pyA, int Nproj);
    
	// 2D Reconstructions
	void ART(float beta);
    void SIRT(float beta);
    void positivity();
    
    // Stochastic Reconstruction
    std::vector<int> calc_proj_order(int n);
    void randART(float beta);
    
    void update_right_slice(Mat *vol);
    void update_left_slice(Mat *vol);
    
	//Forward Project Reconstruction for Data Tolerance Parameter. 
    void forward_projection();
    
    // Acquire local copy of reconstruction.
    void copy_recon();
    
    // Measure 2-norm of projections and reconstruction.
    float matrix_2norm();
    float data_distance();
    float rmse();
    
    // Total variation
    float tv_3D();
    float original_tv_3D();
    void tv_gd_3D(int ng, float dPOCS);
    
    // Return reconstruction to python.
    void gather_recon();
    Mat get_loc_recon(int s);
    
    // Save Recon with MPI Parallel I/O
    void save_recon(char *filename, int type);
    
    // Return projections to python. 
    Mat get_projections();
    int mpi_finalize();
    
    // Set Slices to Zero.
    void restart_recon();
};

#endif /* mpi_ctvlib_hpp */
