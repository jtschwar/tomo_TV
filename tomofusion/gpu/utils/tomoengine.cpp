 //
//  astra_ctlib.cpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "tomoengine.hpp"
#include "regularizers/tv_gd.h"
#include "regularizers/tv_fgp.h"
#include "container/matrix_ops.h"

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <iostream>
#include <cmath>
#include <random>

#include <astra/Float32VolumeData2D.h>
#include <astra/Float32ProjectionData2D.h>
#include <astra/Filters.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace astra;
using namespace Eigen;
using namespace std;
namespace py = pybind11;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;

// Initialize Empty Volume (Useful for Bare Regularization)
tomoengine::tomoengine(int Ns, int Nray) {
    //Intialize all the Member variables.
    Nslice = Ns;
    Ny = Nray;
    Nz = Nray;

    //Initialize all the Slices in Recon as Zero.
    recon = Matrix3D(Ns,Ny,Nz); //Final Reconstruction.
}

// Tomography Constructor
tomoengine::tomoengine(int Ns, int Nray, Vec pyAngles)
{
    //Intialize all the Member variables.
    Nproj = pyAngles.size();
    Nslice = Ns; Ny = Nray; Nz = Nray;
    Nrow = Nray*Nproj;
    Ncol = Ny*Nz;
    
    // Measured and Reprojected Matrices
    b.resize(Nslice, Nrow); g.resize(Nslice, Nrow);
    
    //Initialize all the Slices in Recon as Zero.
    recon = Matrix3D(Ns,Ny,Nz); //Final Reconstruction.

    // INITIALIZE ASTRA OBJECTS.
    
     // Create volume (2D) Geometry
     vol_geom = new CVolumeGeometry2D(Ny,Nz);
    
     // Create Volume ASTRA Object
     vol = new CFloat32VolumeData2D(*vol_geom);
     
     // Specify projection matrix geometries
     float32 *angles = new float32[Nproj];

     for (int j = 0; j < Nproj; j++) {
         angles[j] = pyAngles(j);    }
 
     // Create Projection Matrix Geometry
     proj_geom = new CParallelProjectionGeometry2D(Nproj,Nray,1.0,angles);

     // Create Sinogram ASTRA Object
     sino = new CFloat32ProjectionData2D(*proj_geom);

     // Create CUDA Projector
     proj = new CCudaProjector2D(*proj_geom,*vol_geom);
}

// Set the GPU Index
void tomoengine::set_gpu_id(int id) { 
    gpuID = id;
    recon.gpuIndex = id; }

// Get the GPU Index
int tomoengine::get_gpu_id() { return gpuID; }

// Inialize Initial Volume for Simulation Studies
void tomoengine::initializeInitialVolume() { original_volume = Matrix3D(Nslice,Ny,Nz); }

// Temporary copy for measuring changes in TV and ART.
void tomoengine::initializeReconCopy() { temp_recon =  Matrix3D(Nslice,Ny,Nz); }

//Import tilt series (projections) from Python.
void tomoengine::setTiltSeries(Mat in) { b = in; }

// Import the original volume from python.
void tomoengine::setOriginalVolume(Mat inBuffer, int slice) { original_volume.setData(inBuffer,slice); }

void tomoengine::setRecon(Mat inBuffer, int slice) { recon.setData(inBuffer,slice); }

// Create projections from Volume (for simulation studies)
void tomoengine::create_projections() {
     // Forward Projection Operator
     algo_fp = new CCudaForwardProjectionAlgorithm();

    // int sliceInd;
    for (int s=0; s < Nslice; s++) {
        
        // Pass Input Volume to Astra
        vol->copyData( (float32*) &original_volume.data[original_volume.index(s,0,0)] );

        // Forward Project
        algo_fp->initialize(proj,vol,sino);
        algo_fp->run();

        // Return Sinogram (Measurements) to tomoTV
        memcpy(&b(s,0), sino->getData(), sizeof(float)*Nrow);
    }
}

// Update Proejction Angles
void tomoengine::update_projection_angles(Vec pyAngles) {
    // newNProj = pyAngles.size()
    Nrow = Ny * pyAngles.size();
    b.resize(Nslice, Nrow); g.resize(Nslice, Nrow);
    
    // Specify projection matrix geometries
    float32 *angles = new float32[pyAngles.size()];

    for (int j = 0; j < pyAngles.size(); j++) {
        angles[j] = pyAngles(j);    }
    
    // Delete Previous Projection Matrix Geometry and Projector.
    delete proj_geom, proj, sino;
    
    // Create Projection Matrix Geometry and Projector.
    proj_geom = new CParallelProjectionGeometry2D(pyAngles.size(), Ny, 1, angles);
    proj = new CCudaProjector2D(*proj_geom,*vol_geom);
    sino =  new CFloat32ProjectionData2D(*proj_geom);
    Nproj = pyAngles.size();
}

void tomoengine::initializeSART(std::string order) {
    projOrder = order;
    cout << "ProjectionOrder: " << projOrder << endl;
 
    algo_sart = new CCudaSartAlgorithm();
    algo_sart->initialize(proj,sino,vol);
    algo_sart->setConstraints(true, 0, false, 1);
    if (recon.gpuIndex != -1){algo_sart->setGPUIndex(recon.gpuIndex); }
}

// ART Reconstruction.
void tomoengine::SART(float beta, int nIter) {
    int Nproj = Nrow / Ny;
    algo_sart->updateProjOrder(projOrder);
    
    for (int s=0; s < Nslice; s++) {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA.
        sino->copyData((float32*) &b(s,0));
        vol->copyData( (float32*) &recon.data[recon.index(s,0,0)] );

        // SART Reconstruction
        if (beta != 1) { algo_sart->setRelaxationParameter(beta); }
        algo_sart->updateSlice(sino, vol);
        algo_sart->run(Nproj * nIter);
        
        // Return Slice to tomo_TV
        memcpy(&recon.data[recon.index(s,0,0)], vol->getData(), sizeof(float)*Ny*Nz);
    }
}

void tomoengine::initializeSIRT() {
    algo_sirt = new CCudaSirtAlgorithm();
    algo_sirt->initialize(proj, sino, vol);
    algo_sirt->setConstraints(true, 0, false, 1);
    if (recon.gpuIndex != -1){algo_sirt->setGPUIndex(recon.gpuIndex);}
}

// SIRT Reconstruction.
void tomoengine::SIRT(int nIter) {

    for (int s=0; s < Nslice; s++) {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA
        sino->copyData((float32*) &b(s,0));
        if (momentum) { vol->copyData( (float32*) &yk.data[yk.index(s,0,0)] ); }
        else       { vol->copyData( (float32*) &recon.data[recon.index(s,0,0)] ); }

        // SIRT Reconstruction
        algo_sirt->updateSlice(sino, vol);
        algo_sirt->run(nIter);
        
        // Return Slice to tomo_TV
        if (momentum) { memcpy(&yk.data[yk.index(s,0,0)], vol->getData(), sizeof(float)*Ny*Nz); }
        else       { memcpy(&recon.data[recon.index(s,0,0)], vol->getData(), sizeof(float)*Ny*Nz); }
    }
}

void tomoengine::initializeCGLS() {
    algo_cgls = new CCudaCglsAlgorithm();
    algo_cgls->initialize(proj, sino, vol);
    if (recon.gpuIndex != -1){algo_cgls->setGPUIndex(recon.gpuIndex); }
}

// Conjugate Gradient Least Squares Algorithm
void tomoengine::CGLS(int nIter) {
    for (int s=0; s < Nslice; s++) {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA
        sino->copyData((float32*) &b(s,0));
        vol->copyData( (float32*) &recon.data[recon.index(s,0,0)] );

        // CGLS Reconstruction
        algo_cgls->initialize(proj, sino, vol);
        algo_cgls->run(nIter);
        
        // Return Slice to tomo_TV
        memcpy(&recon.data[recon.index(s,0,0)], vol->getData(), sizeof(float)*Ny*Nz);
    }
    recon.positivity(); // Non-negativity
}

void tomoengine::initializePoissonML() {
    
    // Initialize the Forward and Back-Projection Operators
    initializeFP(); initializeBP();

    // Reshape Intermediate Variables 
    xx.resize(Ny*Nz); Ax.resize(Nrow);
    updateML.resize(xx.size());

    // Estimate Lipschitz Parameter
    Vec cc(xx.size()); cc.setOnes();
    L_Aml = back_projection(forward_projection(cc)).maxCoeff();

    // Normalize the tilt series. 
    if (b.maxCoeff() > 1) { b = b.array() / b.maxCoeff(); }
}

// Initialize the Forward Projection Operator
void tomoengine::initializeFP() { 
    algo_fp = new CCudaForwardProjectionAlgorithm(); 
    algo_fp->initialize(proj,vol,sino);             
    if (recon.gpuIndex != -1){ algo_fp->setGPUIndex(recon.gpuIndex); } 
    outProj.resize(Nrow);          
}

// Forward Projection (ML-Poisson)
Vec tomoengine::forward_projection(const Vec &inVol) {
    // Copy Data to Astra
    vol->copyData((float32*) &inVol(0));

    // Forward Project
    algo_fp->initialize(proj,vol,sino);
    algo_fp->run();

    memcpy(&outProj(0), sino->getData(), sizeof(float)*Nrow);
    return outProj;
}

// Initialize the Back Projection Operator
void tomoengine::initializeBP() { 
    algo_bp = new CCudaBackProjectionAlgorithm(); 
    algo_bp->initialize(proj,sino,vol);
    if (recon.gpuIndex != -1){ algo_bp->setGPUIndex(recon.gpuIndex); } 
    outVol.resize(Ny*Ny);          
}

// Backprojection.
Vec tomoengine::back_projection(const Vec &inProj) {   
    // Copy Data to Astra
    sino->copyData((float32*) &inProj(0));

    // Back Project
    algo_bp->initialize(proj,sino,vol);
    algo_bp->run();

    // Return data from Astra
    memcpy(&outVol(0), vol->getData(), sizeof(float)*Ny*Nz);
    return outVol;
}

float tomoengine::poisson_ML(float lambda) {
    
    float cost = 0; float eps = 1e-1;
    for (int s=0; s<Nslice; s++) {

        memcpy(&xx(0), &recon.data[recon.index(s,0,0)], sizeof(float)*Ny*Nz);

        // Poisson-ML
        Ax = forward_projection(xx);
        updateML = back_projection((Ax - b.row(s).transpose()).array() / (Ax.array() + eps).array() );
        
        // Update along the gradient direction
        xx -= (lambda / L_Aml) * updateML;

        // Return Slice to Reconstruction
        memcpy(&recon.data[recon.index(s,0,0)], &xx(0), sizeof(float)*Ny*Nz);
        
        // Measure Data Fidelity
        cost += ( Ax.array() - b.row(s).transpose().array() * (Ax.array() + eps).log().array() ).sum();  
    }
    recon.positivity();
    return cost; 
}

void tomoengine::initializeFBP(std::string filter) {
    // Possible Inputs for FilterType:
    // none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    // triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    // blackman-nuttall, flat-top, kaiser, parzen
    
   fbfFilter = filter;
   cout << "FBP Filter: " << filter << endl;
   algo_fbp = new CCudaFilteredBackProjectionAlgorithm();
   if (recon.gpuIndex != -1){ algo_fbp->setGPUIndex(recon.gpuIndex); }
}

// Filtered Backprojection.
void tomoengine::FBP(bool apply_positivity) {

    E_FBPFILTER fbfFilt = convertStringToFilter(fbfFilter);
    for (int s=0; s < Nslice; s++) {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA.
        sino->copyData((float32*) &b(s,0));
        vol->copyData((float32*) &recon.data[recon.index(s,0,0)]);

        // FBF Reconstruction
        algo_fbp->initialize(sino,vol,fbfFilt);
        algo_fbp->run();
        
        // Return Slice to tomo_TV
        memcpy(&recon.data[recon.index(s,0,0)], vol->getData(), sizeof(float)*Ny*Nz);
    }
    if (apply_positivity) { recon.positivity(); }
}

// Initialize Additional Reconstruction Volumes for FISTA Algorithm
void tomoengine::initialize_fista() {
    
    // Boolean for FISTA Input / Output
    momentum = true;

    // Initialize Intermediate Variables 
    yk =  Matrix3D(Nslice,Ny,Nz);
    recon_old =  Matrix3D(Nslice,Ny,Nz); 
    
    // Copy Data Respectively
    memcpy(yk.data, recon.data, sizeof(float)*Nslice*Ny*Nz);
    memcpy(recon_old.data, recon.data, sizeof(float)*Nslice*Ny*Nz);

    // Initialize SIRT As Forward Projection
    initializeSIRT(); initializeFP(); initializeBP();

    // Reshape Intermediate Variables 
    xx.resize(Ny*Nz); Ax.resize(Nrow);

    // Initialize Lipschitz Parameter
    Vec cc(Ny*Nz); cc.setOnes();
    L_A = (back_projection(forward_projection(cc))).maxCoeff();  
}

// Return the Estimated Lipschitz Parameter
float tomoengine::get_lipschitz() { return L_A; }

// Set Mometum Bool to False (Return to Regular Tomography)
void tomoengine::remove_momentum() { momentum = false; }

// Nesterov Momentum and Updating Previous Updates
void tomoengine::fista_nesterov_momentum(float beta) { 
    memcpy(recon.data, yk.data, sizeof(float)*Nslice*Ny*Nz);
    cuda_nesterov_momentum(yk.data, recon.data, recon_old.data, beta, Nslice, Ny, Nz);  
    memcpy(recon_old.data, recon.data, sizeof(float)*Nslice*Ny*Nz); }

void tomoengine::least_squares() {
    for (int s=0; s<Nslice; s++){

        if (momentum) { memcpy(&xx(0), &yk.data[yk.index(s,0,0)], sizeof(float)*Ny*Nz); }
        else          { memcpy(&xx(0), &recon.data[recon.index(s,0,0)], sizeof(float)*Ny*Nz); }

        // Gradient Update
        Ax = forward_projection(xx);
        xx -= (1/L_A) * back_projection(Ax - b.row(s).transpose());

        // Return Slice and Measure Cost Function
        if (momentum) { memcpy(&yk.data[yk.index(s,0,0)], &xx(0), sizeof(float)*Ny*Nz); }
        else          { memcpy(&recon.data[recon.index(s,0,0)], &xx(0), sizeof(float)*Ny*Nz);  }     
    }
    // return cost;
}

// Create Local Copy of Reconstruction. 
void tomoengine::copy_recon() { memcpy(temp_recon.data, recon.data, sizeof(float)*Nslice*Ny*Nz); }

// Measure the 2 norm between temporary and current reconstruction.
float tomoengine::matrix_2norm() { return sqrt(cuda_euclidean_dist(recon.data, temp_recon.data, Nslice, Ny, Nz)); }

// Measure the 2 norm between experimental and reconstructed projections.
float tomoengine::data_distance() {
    forwardProjection();
    return (g - b).norm();          
}

// Foward project the data.
void tomoengine::forwardProjection() {
    for (int s=0; s < Nslice; s++) {
        vol->copyData((float32*) &recon.data[recon.index(s,0,0)]);

        // Forward Project
        algo_fp->updateSlice(sino, vol);
        // algo_fp->initialize(proj,vol,sino);
        algo_fp->run();
        
        memcpy(&g(s,0), sino->getData(), sizeof(float)*Nrow);
    }
}

// Measure the RMSE (simulation studies)
float tomoengine::rmse() { return sqrt(cuda_rmse(recon.data, original_volume.data, Nslice, Ny, Nz) / (Nslice * Ny * Nz)); }

// Measure Reconstruction's L1 Norm
float tomoengine::l1_norm() { return recon.l1_norm(); }

// Soft Threshold Operator
void tomoengine::soft_threshold(float lambda){ 
    if (momentum) { yk.soft_threshold(lambda); yk.positivity(); }
    else          { recon.soft_threshold(lambda); recon.positivity(); }  }

//Measure Reconstruction's TV.
float tomoengine::tv_3D() { return cuda_tv_3D(recon.data, Nslice, Ny, Nz); }

//Measure Original Volume's TV.
float tomoengine::original_tv_3D() { return cuda_tv_3D(original_volume.data, Nslice, Ny, Nz); }

// TV Minimization (Gradient Descent)
float tomoengine::tv_gd_3D(int ng, float dPOCS) { return cuda_tv_gd_3D(recon.data, ng, dPOCS, Nslice, Ny, Nz); }

// TV Minimization (Gradient Projection Method)
float tomoengine::tv_fgp_3D(int ng, float lambda) {  return cuda_tv_fgp_3D(recon.data, ng, lambda, Nslice, Ny, Nz); }

// Return Reconstruction to Python.
Mat tomoengine::getRecon(int slice) { return recon.getData(slice); }

//Return the Experimental projections.
Mat tomoengine::get_projections() { return b; }

// Return the Model Reprojections from the Volume.
Mat tomoengine::get_model_projections() { return g; }

// Restart the Reconstruction (Reset to Zero). 
void tomoengine::restart_recon() { 
    memset(recon.data, 0, sizeof(float)*Nslice*Ny*Nz); 
    if (momentum) {
        memset(yk.data, 0, sizeof(float)*Nslice*Ny*Nz);
        memset(recon_old.data, 0, sizeof(float)*Nslice*Ny*Nz); 
    }
}

// Add poisson noise to projections.
void tomoengine::poissonNoise(int Nc) {
    Mat temp_b = b;
    float mean = b.mean();
    float N = b.sum();
    b  = b / ( b.sum() ) * Nc * b.size();
    std::default_random_engine generator;
    for(int i=0; i < b.size(); i++)
    {
       std::poisson_distribution<int> distribution(b(i));
       b(i) = distribution(generator);
       
    }
    b = b / ( Nc * b.size() ) * N;
}

//Python functions for tomoengine module.
PYBIND11_MODULE(tomoengine, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions using ASTRA Cuda Library";
    py::class_<tomoengine> tomoengine(m, "tomoengine");
    tomoengine.def(py::init<int,int>());
    tomoengine.def(py::init<int,int,Vec>());
    tomoengine.def("initialize_initial_volume", &tomoengine::initializeInitialVolume, "Initialize Original Data");
    tomoengine.def("initialize_recon_copy", &tomoengine::initializeReconCopy, "Initalize Copy Data of Recon");
    tomoengine.def("set_tilt_series", &tomoengine::setTiltSeries, "Pass the Projections to C++ Object");
    tomoengine.def("set_original_volume", &tomoengine::setOriginalVolume, "Pass the Volume to C++ Object");
    tomoengine.def("update_projection_angles", &tomoengine::update_projection_angles, "Update Projection Angles");
    tomoengine.def("create_projections", &tomoengine::create_projections, "Create Projections from Volume");
    tomoengine.def("get_recon", &tomoengine::getRecon, "Return the Reconstruction to Python");
    tomoengine.def("set_recon", &tomoengine::setRecon, "Return the Reconstruction to Python");
    tomoengine.def("get_gpu_id", &tomoengine::get_gpu_id, "Get the GPU ID");
    tomoengine.def("set_gpu", &tomoengine::set_gpu_id, "Set the GPU ID");
    tomoengine.def("initialize_SART", &tomoengine::initializeSART, "Initialize SART");
    tomoengine.def("SART", &tomoengine::SART, "ART Reconstruction");
    tomoengine.def("initialize_SIRT", &tomoengine::initializeSIRT, "Initialize SIRT");
    tomoengine.def("SIRT", &tomoengine::SIRT, "SIRT Reconstruction");
    tomoengine.def("initialize_FBP", &tomoengine::initializeFBP, "Initialize Filtered BackProjection");
    tomoengine.def("FBP", &tomoengine::FBP, "Filtered Backprojection");
    tomoengine.def("initialize_FP", &tomoengine::initializeFP, "Initialize Forward Projection");
    tomoengine.def("initialize_BP", &tomoengine::initializeBP, "Initialize Back Projection");
    tomoengine.def("initialize_CGLS", &tomoengine::initializeCGLS, "Initialize Conjugate Gradient Method for Least Squares");
    tomoengine.def("CGLS", &tomoengine::CGLS, "Conjugate Gradient Method for Least Squares");
    tomoengine.def("initialize_poisson_ML", &tomoengine::initializePoissonML, "Poisson ML Reconstruction");    
    tomoengine.def("poisson_ML", &tomoengine::poisson_ML, "Poisson ML Reconstruction");
    tomoengine.def("forward_projection", &tomoengine::forwardProjection, "Forward Projection");
    tomoengine.def("soft_threshold",&tomoengine::soft_threshold,"Soft Thresholding Operator");
    tomoengine.def("l1_norm",&tomoengine::l1_norm, "L1 Norm");
    tomoengine.def("initialize_fista", &tomoengine::initialize_fista, "Initialize FISTA");
    tomoengine.def("get_lipschitz", &tomoengine::get_lipschitz, "Get the Lipschitz Parameter");
    tomoengine.def("fista_momentum", &tomoengine::fista_nesterov_momentum,"Fista Momentum Acceleration");
    tomoengine.def("remove_momentum", &tomoengine::remove_momentum,"Remove Momentum from Optimization");
    tomoengine.def("copy_recon", &tomoengine::copy_recon, "Copy the reconstruction");
    tomoengine.def("matrix_2norm", &tomoengine::matrix_2norm, "Calculate L2-Norm of Reconstruction");
    tomoengine.def("data_distance", &tomoengine::data_distance, "Calculate L2-Norm of Projection (aka Vectors)");
    tomoengine.def("rmse", &tomoengine::rmse, "Calculate reconstruction's RMSE");
    tomoengine.def("tv", &tomoengine::tv_3D, "Measure TV of Reconstruction");
    tomoengine.def("original_tv", &tomoengine::original_tv_3D, "Measure original TV");
    tomoengine.def("tv_gd", &tomoengine::tv_gd_3D, "3D TV Gradient Descent");
    tomoengine.def("tv_fgp", &tomoengine::tv_fgp_3D, "3D TV Fast Gradient Projection");
    tomoengine.def("get_projections", &tomoengine::get_projections, "Return the projection matrix to python");
    tomoengine.def("get_model_projections", &tomoengine::get_model_projections, "Return the re-projection matrix to python");
    tomoengine.def("poisson_noise", &tomoengine::poissonNoise, "Add Poisson Noise to Projections");
    tomoengine.def("restart_recon", &tomoengine::restart_recon, "Set all the Slices Equal to Zero");
}
