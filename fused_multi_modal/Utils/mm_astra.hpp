//
//  astra_ctlib.hpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#ifndef mm_astra_hpp
#define mm_astra_hpp

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "container/Matrix4D.h"


#include <astra/Float32VolumeData2D.h>
#include <astra/Float32ProjectionData2D.h>

#include <astra/VolumeGeometry2D.h>

#include <astra/ProjectionGeometry2D.h>
#include <astra/ParallelProjectionGeometry2D.h>
#include <astra/CudaProjector2D.h>

#include <astra/CudaForwardProjectionAlgorithm.h>
#include <astra/CudaBackProjectionAlgorithm.h>
#include <astra/CudaFilteredBackProjectionAlgorithm.h>
#include <astra/CudaSartAlgorithm.h>
#include <astra/CudaSirtAlgorithm.h>

#include <vector>

using namespace astra;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;


class mm_astra
{

public: 

    // Eigen - tomoTV Member Variables.
    Matrix4D recon, original_volume, temp_recon;
    int NrowChem, NrowHaadf, Ncol, Nslice, Ny, Nz, Nel;
    int NprojHaadf, NprojChem, nPix; 
    int gpuID = -1;

    // Intermediate Reconstruction Variables
    Vec xx, Ax, updateCHEM, updateHAADF, modelHAADF, updateVol;
    Vec outProj, outVol, outProj4D, outVol4D;

    // Raw Data and Reprojection
    Mat bh, bChem, g, gChem;
    
    // Summation matrix
    Eigen::SparseMatrix<float, Eigen::RowMajor> sigma, spXXmatrix;

    // Inialize measurement to false. 
    bool measureHaadf = false; bool measureChem = false;
    float gamma = 1; float eps = 1e-1;
    float L_Aps, L_ASig;
    
    // Astra Member Variables
    
    // Volume Geometry and Object
    CVolumeGeometry2D *vol_geom;
    CFloat32VolumeData2D *vol;
    
    // Projection / Sinogram Geometry and Object
    CParallelProjectionGeometry2D *haadfProjGeom, *chemProjGeom;
    CFloat32ProjectionData2D *haadfSino, *chemSino;
    
    // Cuda Projector and Forward Projection Operator. 
    CCudaProjector2D *hProj, *cProj;
    CCudaForwardProjectionAlgorithm *algo_fp;
    CCudaBackProjectionAlgorithm *algo_bp;
    CCudaSartAlgorithm *algo_sart;
    CCudaSirtAlgorithm *algo_sirt;
    CCudaFilteredBackProjectionAlgorithm *algo_fbp;

    // Auxilary variables for Sart and FBP.
    std::string projOrder; std::string fbfFilter; 
 
	// Initializes Measurement Matrix. 
	mm_astra(int Nslice, int Nray, int Nelements);
    mm_astra(int Nslice, int Nray, int Nelements, Vec haadfAngles, Vec chemAngles);
    void initialize_initial_volume();

    // Access Member variables (measureHaadf,measureChem,eps)
    bool get_measureHaadf();  void set_measureHaadf(bool inBool);
    bool get_measureChem();   void set_measureChem(bool inBool);
    float get_gamma();        void set_gamma(float inGamma);
    float get_eps();          void set_eps(float inEps);
    int get_gpu_id();         void set_gpu_id(int id);


	// Initialize Experimental Projections. 
	void set_haadf_tilt_series(Mat in);
    void set_chem_tilt_series(Mat in);

	// Compute Lipschitz Constant (SIRT Reconstruction).
    void estimate_lipschitz();
    
	//Forward Project Reconstruction for Data Tolerance Parameter. 
	Vec forward_projection(const Vec &inVol);
    Vec forward_projection4D(const Vec &inVol);
    Vec back_projection(const Vec &inProj);
    Vec back_projection4D(const Vec &inProj);

    float data_distance();

    void load_summation_matrix(Mat pySig);

    float poisson_ml(float lambdaCHEM);
    // std::tuple<float,float> data_fusion(float lambdaHAADF, float lambdaCHEM);
    void rescale_projections();

    // Acquire local copy of reconstruction.
    void copy_recon();
    
    // Total variation Regularization
    float tv_gd_4D(int ng, float dPOCS);
    float tv_fgp_4D(int ng, float lambda);

    // Generate Config File for Reconstruction Operators
    void initializeSIRT();
    void initializeSART(std::string projOrder);
    void initializeFBP(std::string filter);
    void initializeFP();
    void initializeBP();
    
	// 2D Reconstructions
    Vec SART(const Vec &inVol,int s,int nIter);
    void FBP(bool apply_positivity);

    // Non-Multi-Modal Reconstructions
    void chemical_SART(int nIter); void chemical_SIRT(int nIter);

    // Call Data Fusion
        Vec fuse(const Vec &inVol, int s, int nIter, std::string method);
    std::tuple<float,float> call_sirt_data_fusion(float lammbdaHAADF, float lambdaCHEM, int nIter);
    std::tuple<float,float> call_sart_data_fusion(float lammbdaHAADF, float lambdaCHEM);
    std::tuple<float,float> data_fusion(float lambdaHAADF, float lambdaCHEM, int nIter, std::string method);

    // void SIRT(int nIter=1);
    // error: no matching function for call to ‘mm_astra::SIRT(astra::CFloat32ProjectionData2D*&, Matrix4D&, int, int&, int&)’ 
    Vec SIRT(const Vec &inVol,int s,int nIter);
    void SIRT(int e, int s, int nIter);


    void SART(int e, int s, int nIter);    



    std::tuple<float,float> SIRT_data_fusion(float lambdaHAADF, float lambdaCHEM, int nIter);
    std::tuple<float,float> SART_data_fusion(float lambdaHAADF, float lambdaCHEM, int nIter);

    // Measure RMSE
    Vec rmse();
    // float tv_3D(int NelSel);
    // void print_recon();

    // Set Slices to Zero.
    void restart_recon();
    
    // Return reconstruction to python.
    void set_original_volume(Mat inBuffer, int slice, int element);
    void set_recon(Mat inBuffer, int slice, int element);
    Mat get_recon(int slice,int element);
    Mat get_gt(int element, int slice);
    void save_recon(char *filename);
    
    // Return projections to python
    Mat get_haadf_projections(); Mat get_chem_projections();
    Mat get_model_projections();
};

#endif /* astra_ctlib.hpp */
