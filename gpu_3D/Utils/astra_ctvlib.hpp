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

#include <astra/CudaBackProjectionAlgorithm.h>
#include <astra/CudaForwardProjectionAlgorithm.h>
#include <astra/CudaFilteredBackProjectionAlgorithm.h>
#include <astra/CudaSartAlgorithm.h>
#include <astra/CudaSirtAlgorithm.h>
#include <astra/CudaCglsAlgorithm.h>

using namespace astra;

class astra_ctvlib
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;

public: 

    // Eigen - tomoTV Member Variables.
    Matrix3D recon, temp_recon, original_volume, yk, recon_old;
    int Nrow, Ncol, Nslice, Nproj, Ny, Nz;
    int gpuID = -1;

    // Intermediate Reconstruction Variables
    Eigen::VectorXf innerProduct, updateML, xx, Ax, outProj, outVol;
    float L_Aml, L_A;

    // Raw Data and Reprojection
    Mat b, g;

    bool momentum = false;
    
    // Astra Member Variables
    
    // Volume Geometry and Object
    CVolumeGeometry2D *vol_geom;                CFloat32VolumeData2D *vol;
    
    // Projection / Sinogram Geometry and Object
    CParallelProjectionGeometry2D *proj_geom;   CFloat32ProjectionData2D *sino;
    
    // Cuda Projector and Forward Projection Operator. 
    CCudaForwardProjectionAlgorithm *algo_fp;   CCudaBackProjectionAlgorithm *algo_bp;
    CCudaSartAlgorithm *algo_sart;              CCudaSirtAlgorithm *algo_sirt;
    CCudaProjector2D *proj;                     CCudaCglsAlgorithm *algo_cgls;
    CCudaFilteredBackProjectionAlgorithm *algo_fbp;

    // Auxilary variables for Sart and FBP.
    std::string fbfFilter;                      std::string projOrder;
 
	// Initializes Measurement Matrix. 
	astra_ctvlib(int Nslice, int Nray);
    astra_ctvlib(int Nslice, int Nray, Vec pyAngles);
    void initializeInitialVolume();             void initializeReconCopy();

	// Initialize Experimental Projections. 
	void setTiltSeries(Mat in);
    void setOriginalVolume(Mat in, int slice);
    int get_gpu_id();                           void set_gpu_id(int id);

    // Create Projections and Add Poisson Noise (Simulations)
    void create_projections();                  void poissonNoise(int SNR);

	// Compute Lipschitz Constant (SIRT Reconstruction).
    // void lipschitz();
    
    // Generate Config File for Reconstruction Operators
    void initializeSIRT();                      void initializeSART(std::string projOrder);
    void initializeFBP(std::string filter);     void initializeCGLS();
    void initializeFP();                        void initializeBP();
    void initializePoissonML();
    
	// 2D Reconstructions
    void update_projection_angles(Vec pyAngles);
    void SART(float beta, int nIter=1);         void SIRT(int nIter=1);
    void FBP(bool apply_positivity=true);       void CGLS(int nIter=1);

    // FISTA Reconstruction
    void initialize_fista();                    void remove_momentum();    
    void soft_threshold(float lambda);          void fista_nesterov_momentum(float beta);
    void initialize_ogm();                      float get_lipschitz();
    void least_squares();

    // Forward And Back-Projection Operator
    Vec forward_projection(const Vec &inVol);   Vec back_projection(const Vec &inProj);

    // Reconstruct with Maximum-Likelihood (KL-Divergence)
    float poisson_ML(float lambda);

	//Forward Project Reconstruction for Data Tolerance Parameter. 
	void forwardProjection();

    // Acquire local copy of reconstruction.
    void copy_recon();
    
    // Measure 2-norm of projections and reconstruction.
    float matrix_2norm();                       float data_distance();
    float l1_norm();                            float rmse();
    
    // Total variation
    float tv_3D();                              float original_tv_3D();
    float tv_gd_3D(int ng, float dPOCS);        float tv_fgp_3D(int ng, float lambda);

    // Set Slices to Zero.
    void restart_recon();
    
    // Return reconstruction to python.
    void setRecon(Mat in, int s);               Mat getRecon(int i);
    void save_recon(char *filename);
    
    // Return projections to python
    Mat get_projections();                      Mat get_model_projections();
};

#endif /* astra_ctlib.hpp */
