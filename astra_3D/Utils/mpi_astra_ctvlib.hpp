//
//  mpi_astra_ctlib.hpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef mpi_astra_ctvlib_hpp
#define mpi_astra_ctvlib_hpp

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "container/Matrix3D.h"

#include <astra/Float32VolumeData2D.h>
#include <astra/Float32ProjectionData2D.h>

#include <astra/VolumeGeometry2D.h>

#include <astra/ProjectionGeometry2D.h>
#include <astra/ParallelProjectionGeometry2D.h>
#include <astra/CudaProjector2D.h>

#include <astra/CudaForwardProjectionAlgorithm.h>
#include <astra/CudaFilteredBackProjectionAlgorithm.h>
#include <astra/CudaSartAlgorithm.h>
#include <astra/CudaSirtAlgorithm.h>

using namespace astra;

class mpi_astra_ctvlib
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;

public: 

    // Eigen - tomoTV Member Variables.
    Matrix3D recon, temp_recon, original_volume;
    int Nrow, Ncol, Nslice, Nslice_loc, Ny, Nz, nproc, rank, size;
    int first_slice, last_slice, nDevices, localDevice;
    Mat b, g;
    
    // Astra Member Variables
    
    // Volume Geometry and Object
    CVolumeGeometry2D *vol_geom;
    CFloat32VolumeData2D *vol;
    
    // Projection / Sinogram Geometry and Object
    CParallelProjectionGeometry2D *proj_geom;
    CFloat32ProjectionData2D *sino;
    
    // Cuda Projector and Forward Projection Operator. 
    CCudaProjector2D *proj;
    CCudaForwardProjectionAlgorithm *algo_fp;
    CCudaSartAlgorithm *algo_sart;
    CCudaSirtAlgorithm *algo_sirt;
    CCudaFilteredBackProjectionAlgorithm *algo_fbp;

    // Auxilary variables for Sart and FBP.
    std::string fbfFilter;
    std::string projOrder;

	// Initializes Measurement Matrix. 
	mpi_astra_ctvlib(int Nslice, int Nray, int Nproj, Vec pyAngles);
    int get_rank();
    int get_nproc();
    int get_Nslice_loc();
    int get_first_slice();
    void initilizeInitialVolume();
    void initializeReconCopy();
    void checkNumGPUs();

	// Initialize Experimental Projections. 
	void setTiltSeries(Mat in);
    void setRecon(Mat in, int s);
    void setOriginalVolume(Mat in, int slice);
    
    // Create Projections and Add Poisson Noise (Simulations)
    void create_projections();
    void poissonNoise(int SNR);
    
    // Generate Config File for Reconstruction Operators
    void initializeSIRT();
    void initializeSART(std::string projOrder);
    void initializeFBP(std::string filter);

	// 2D Reconstructions
    void update_projection_angles(int Nproj, Vec pyAngles);
    void SART(float beta);
    void SIRT();
    void FBP();
    void positivity();
    
	//Forward Project Reconstruction for Data Tolerance Parameter. 
	void forwardProjection();

    // Acquire local copy of reconstruction.
    void copy_recon();
    
    // Measure 2-norm of projections and reconstruction.
    float matrix_2norm();
    float vector_2norm();
    float dyn_vector_2norm(int dyn_ind);
    float rmse();
    
    // Total variation
    void updateRightSlice(Matrix3D vol);
    void updateLeftSlice(Matrix3D vol);
    float tv_3D();
    float original_tv_3D();
    float tv_gd_3D(int ng, float dPOCS);
    
    // Set Slices to Zero.
    void restart_recon();
    
    // Return reconstruction to python.
    Mat getRecon(int i);
    
    // Return projections to python. 
    Mat get_projections();
    Mat get_model_projections();
    
};

#endif /* mpi_astra_ctlib.hpp */
