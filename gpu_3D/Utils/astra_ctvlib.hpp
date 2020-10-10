//
//  astra_ctlib.hpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef astra_ctvlib_hpp
#define astra_ctvlib_hpp

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

class astra_ctvlib
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;

public: 

    // Eigen - tomoTV Member Variables.
    Matrix3D recon, temp_recon, original_volume;
    int Nrow, Ncol, Nslice, Ny, Nz;
    Eigen::VectorXf innerProduct;
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
	astra_ctvlib(int Nslice, int Nray, int Nproj, Vec pyAngles);
    void initilizeInitialVolume();
    void initializeReconCopy();

	// Initialize Experimental Projections. 
	void setTiltSeries(Mat in);
    void setOriginalVolume(Mat in, int slice);
    
    // Create Projections and Add Poisson Noise (Simulations)
    void create_projections();
    void poissonNoise(int SNR);

	// Compute Lipschitz Constant (SIRT Reconstruction).
    float lipschits();
    
    // Generate Config File for Reconstruction Operators
    void initializeSIRT();
    void initializeSART(std::string projOrder);
    void initializeFBP(std::string filter);
    void initializeFP();
    
	// 2D Reconstructions
    void update_projection_angles(int Nproj, Vec pyAngles);
    void SART(float beta, int nIter=1);
    void SIRT(int nIter=1);
    void FBP(bool apply_positivity);
    
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
    float tv_3D();
    float original_tv_3D();
    float tv_gd_3D(int ng, float dPOCS);
    float tv_fgp_3D(int ng, float lambda);

    // Set Slices to Zero.
    void restart_recon();
    
    // Return reconstruction to python.
    void setRecon(Mat in, int s);
    Mat getRecon(int i);
    
    // Return projections to python
    Mat get_projections();
    Mat get_model_projections();
};

#endif /* astra_ctlib.hpp */
