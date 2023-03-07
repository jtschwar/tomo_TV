 //
//  astra_ctlib.cpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "astra_ctvlib.hpp"
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
astra_ctvlib::astra_ctvlib(int Ns, int Nray) {
    //Intialize all the Member variables.
    Nslice = Ns;
    Ny = Nray;
    Nz = Nray;

    //Initialize all the Slices in Recon as Zero.
    recon = Matrix3D(Ns,Ny,Nz); //Final Reconstruction.
}

// Tomography Constructor
astra_ctvlib::astra_ctvlib(int Ns, int Nray, Vec pyAngles)
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
     vol = new CFloat32VolumeData2D(vol_geom);
     
     // Specify projection matrix geometries
     float32 *angles = new float32[Nproj];

     for (int j = 0; j < Nproj; j++) {
         angles[j] = pyAngles(j);    }
 
     // Create Projection Matrix Geometry
     proj_geom = new CParallelProjectionGeometry2D(Nproj,Nray,1.0,angles);

     // Create Sinogram ASTRA Object
     sino = new CFloat32ProjectionData2D(proj_geom);

     // Create CUDA Projector
     proj = new CCudaProjector2D(proj_geom,vol_geom);
}

// Set the GPU Index
void astra_ctvlib::set_gpu_id(int id) { 
    gpuID = id;
    recon.gpuIndex = id; }

// Get the GPU Index
int astra_ctvlib::get_gpu_id() { return gpuID; }

// Inialize Initial Volume for Simulation Studies
void astra_ctvlib::initializeInitialVolume() { original_volume = Matrix3D(Nslice,Ny,Nz); }

// Temporary copy for measuring changes in TV and ART.
void astra_ctvlib::initializeReconCopy() { temp_recon =  Matrix3D(Nslice,Ny,Nz); }

//Import tilt series (projections) from Python.
void astra_ctvlib::setTiltSeries(Mat in) { b = in; }

// Import the original volume from python.
void astra_ctvlib::setOriginalVolume(Mat inBuffer, int slice) { original_volume.setData(inBuffer,slice); }

void astra_ctvlib::setRecon(Mat inBuffer, int slice) { recon.setData(inBuffer,slice); }

// Create projections from Volume (for simulation studies)
void astra_ctvlib::create_projections() {
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
void astra_ctvlib::update_projection_angles(Vec pyAngles) {
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
    proj = new CCudaProjector2D(proj_geom,vol_geom);
    sino =  new CFloat32ProjectionData2D(proj_geom);
    Nproj = pyAngles.size();
}

void astra_ctvlib::initializeSART(std::string order) {
    projOrder = order;
    cout << "ProjectionOrder: " << projOrder << endl;
 
    algo_sart = new CCudaSartAlgorithm();
    algo_sart->initialize(proj,sino,vol);
    algo_sart->setConstraints(true, 0, false, 1);
    if (recon.gpuIndex != -1){algo_sart->setGPUIndex(recon.gpuIndex); }
}

// ART Reconstruction.
void astra_ctvlib::SART(float beta, int nIter) {
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

void astra_ctvlib::initializeSIRT() {
    algo_sirt = new CCudaSirtAlgorithm();
    algo_sirt->initialize(proj, sino, vol);
    algo_sirt->setConstraints(true, 0, false, 1);
    if (recon.gpuIndex != -1){algo_sirt->setGPUIndex(recon.gpuIndex);}
}

// SIRT Reconstruction.
void astra_ctvlib::SIRT(int nIter) {

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

void astra_ctvlib::initializeCGLS() {
    algo_cgls = new CCudaCglsAlgorithm();
    algo_cgls->initialize(proj, sino, vol);
    if (recon.gpuIndex != -1){algo_cgls->setGPUIndex(recon.gpuIndex); }
}

// Conjugate Gradient Least Squares Algorithm
void astra_ctvlib::CGLS(int nIter) {
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

void astra_ctvlib::initializePoissonML() {
    
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
void astra_ctvlib::initializeFP() { 
    algo_fp = new CCudaForwardProjectionAlgorithm(); 
    algo_fp->initialize(proj,vol,sino);             
    if (recon.gpuIndex != -1){ algo_fp->setGPUIndex(recon.gpuIndex); } 
    outProj.resize(Nrow);          
}

// Forward Projection (ML-Poisson)
Vec astra_ctvlib::forward_projection(const Vec &inVol) {
    // Copy Data to Astra
    vol->copyData((float32*) &inVol(0));

    // Forward Project
    algo_fp->initialize(proj,vol,sino);
    algo_fp->run();

    memcpy(&outProj(0), sino->getData(), sizeof(float)*Nrow);
    return outProj;
}

// Initialize the Back Projection Operator
void astra_ctvlib::initializeBP() { 
    algo_bp = new CCudaBackProjectionAlgorithm(); 
    algo_bp->initialize(proj,sino,vol);
    if (recon.gpuIndex != -1){ algo_bp->setGPUIndex(recon.gpuIndex); } 
    outVol.resize(Ny*Ny);          
}

// Backprojection.
Vec astra_ctvlib::back_projection(const Vec &inProj) {   
    // Copy Data to Astra
    sino->copyData((float32*) &inProj(0));

    // Back Project
    algo_bp->initialize(proj,sino,vol);
    algo_bp->run();

    // Return data from Astra
    memcpy(&outVol(0), vol->getData(), sizeof(float)*Ny*Nz);
    return outVol;
}

float astra_ctvlib::poisson_ML(float lambda) {
    
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

void astra_ctvlib::initializeFBP(std::string filter) {
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
void astra_ctvlib::FBP(bool apply_positivity) {

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
void astra_ctvlib::initialize_fista() {
    
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
float astra_ctvlib::get_lipschitz() { return L_A; }

// Set Mometum Bool to False (Return to Regular Tomography)
void astra_ctvlib::remove_momentum() { momentum = false; }

// Nesterov Momentum and Updating Previous Updates
void astra_ctvlib::fista_nesterov_momentum(float beta) { 
    memcpy(recon.data, yk.data, sizeof(float)*Nslice*Ny*Nz);
    cuda_nesterov_momentum(yk.data, recon.data, recon_old.data, beta, Nslice, Ny, Nz);  
    memcpy(recon_old.data, recon.data, sizeof(float)*Nslice*Ny*Nz); }

void astra_ctvlib::least_squares() {
    float cost = 0;
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
    return cost;
}

// Create Local Copy of Reconstruction. 
void astra_ctvlib::copy_recon() { memcpy(temp_recon.data, recon.data, sizeof(float)*Nslice*Ny*Nz); }

// Measure the 2 norm between temporary and current reconstruction.
float astra_ctvlib::matrix_2norm() { return sqrt(cuda_euclidean_dist(recon.data, temp_recon.data, Nslice, Ny, Nz)); }

// Measure the 2 norm between experimental and reconstructed projections.
float astra_ctvlib::data_distance() {
    forwardProjection();
    return (g - b).norm() / g.size();  } // Nrow*Nslice,sum_{ij} M_ij^2 / Nrow*Nslice
// return (g - b).norm();          }

// Foward project the data.
void astra_ctvlib::forwardProjection() {
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
float astra_ctvlib::rmse() { return sqrt(cuda_rmse(recon.data, original_volume.data, Nslice, Ny, Nz) / (Nslice * Ny * Nz)); }

// Measure Reconstruction's L1 Norm
float astra_ctvlib::l1_norm() { return recon.l1_norm(); }

// Soft Threshold Operator
void astra_ctvlib::soft_threshold(float lambda){ 
    if (momentum) { yk.soft_threshold(lambda); yk.positivity(); }
    else          { recon.soft_threshold(lambda); recon.positivity(); }  }

//Measure Reconstruction's TV.
float astra_ctvlib::tv_3D() { return cuda_tv_3D(recon.data, Nslice, Ny, Nz); }

//Measure Original Volume's TV.
float astra_ctvlib::original_tv_3D() { return cuda_tv_3D(original_volume.data, Nslice, Ny, Nz); }

// TV Minimization (Gradient Descent)
float astra_ctvlib::tv_gd_3D(int ng, float dPOCS) { return cuda_tv_gd_3D(recon.data, ng, dPOCS, Nslice, Ny, Nz); }

// TV Minimization (Gradient Projection Method)
float astra_ctvlib::tv_fgp_3D(int ng, float lambda) {  return cuda_tv_fgp_3D(recon.data, ng, lambda, Nslice, Ny, Nz); }

// Return Reconstruction to Python.
Mat astra_ctvlib::getRecon(int slice) { return recon.getData(slice); }

//Return the Experimental projections.
Mat astra_ctvlib::get_projections() { return b; }

// Return the Model Reprojections from the Volume.
Mat astra_ctvlib::get_model_projections() { return g; }

// Restart the Reconstruction (Reset to Zero). 
void astra_ctvlib::restart_recon() { 
    memset(recon.data, 0, sizeof(float)*Nslice*Ny*Nz); 
    if (momentum) {
        memset(yk.data, 0, sizeof(float)*Nslice*Ny*Nz);
        memset(recon_old.data, 0, sizeof(float)*Nslice*Ny*Nz); 
    }
}

// Add poisson noise to projections.
void astra_ctvlib::poissonNoise(int Nc) {
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

//Python functions for astra_ctvlib module.
PYBIND11_MODULE(astra_ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions using ASTRA Cuda Library";
    py::class_<astra_ctvlib> astra_ctvlib(m, "astra_ctvlib");
    astra_ctvlib.def(py::init<int,int>());
    astra_ctvlib.def(py::init<int,int,Vec>());
    astra_ctvlib.def("initialize_initial_volume", &astra_ctvlib::initializeInitialVolume, "Initialize Original Data");
    astra_ctvlib.def("initialize_recon_copy", &astra_ctvlib::initializeReconCopy, "Initalize Copy Data of Recon");
    astra_ctvlib.def("set_tilt_series", &astra_ctvlib::setTiltSeries, "Pass the Projections to C++ Object");
    astra_ctvlib.def("set_original_volume", &astra_ctvlib::setOriginalVolume, "Pass the Volume to C++ Object");
    astra_ctvlib.def("update_projection_angles", &astra_ctvlib::update_projection_angles, "Update Projection Angles");
    astra_ctvlib.def("create_projections", &astra_ctvlib::create_projections, "Create Projections from Volume");
    astra_ctvlib.def("get_recon", &astra_ctvlib::getRecon, "Return the Reconstruction to Python");
    astra_ctvlib.def("set_recon", &astra_ctvlib::setRecon, "Return the Reconstruction to Python");
    astra_ctvlib.def("get_gpu_id", &astra_ctvlib::get_gpu_id, "Get the GPU ID");
    astra_ctvlib.def("set_gpu", &astra_ctvlib::set_gpu_id, "Set the GPU ID");
    astra_ctvlib.def("initialize_SART", &astra_ctvlib::initializeSART, "Initialize SART");
    astra_ctvlib.def("SART", &astra_ctvlib::SART, "ART Reconstruction");
    astra_ctvlib.def("initialize_SIRT", &astra_ctvlib::initializeSIRT, "Initialize SIRT");
    astra_ctvlib.def("SIRT", &astra_ctvlib::SIRT, "SIRT Reconstruction");
    astra_ctvlib.def("initialize_FBP", &astra_ctvlib::initializeFBP, "Initialize Filtered BackProjection");
    astra_ctvlib.def("FBP", &astra_ctvlib::FBP, "Filtered Backprojection");
    astra_ctvlib.def("initialize_FP", &astra_ctvlib::initializeFP, "Initialize Forward Projection");
    astra_ctvlib.def("initialize_BP", &astra_ctvlib::initializeBP, "Initialize Back Projection");
    astra_ctvlib.def("initialize_CGLS", &astra_ctvlib::initializeCGLS, "Initialize Conjugate Gradient Method for Least Squares");
    astra_ctvlib.def("CGLS", &astra_ctvlib::CGLS, "Conjugate Gradient Method for Least Squares");
    astra_ctvlib.def("initialize_poisson_ML", &astra_ctvlib::initializePoissonML, "Poisson ML Reconstruction");    
    astra_ctvlib.def("poisson_ML", &astra_ctvlib::poisson_ML, "Poisson ML Reconstruction");
    astra_ctvlib.def("forward_projection", &astra_ctvlib::forwardProjection, "Forward Projection");
    astra_ctvlib.def("soft_threshold",&astra_ctvlib::soft_threshold,"Soft Thresholding Operator");
    astra_ctvlib.def("l1_norm",&astra_ctvlib::l1_norm, "L1 Norm");
    astra_ctvlib.def("initialize_fista", &astra_ctvlib::initialize_fista, "Initialize FISTA");
    astra_ctvlib.def("get_lipschitz", &astra_ctvlib::get_lipschitz, "Get the Lipschitz Parameter");
    astra_ctvlib.def("fista_momentum", &astra_ctvlib::fista_nesterov_momentum,"Fista Momentum Acceleration");
    astra_ctvlib.def("remove_momentum", &astra_ctvlib::remove_momentum,"Remove Momentum from Optimization");
    astra_ctvlib.def("copy_recon", &astra_ctvlib::copy_recon, "Copy the reconstruction");
    astra_ctvlib.def("matrix_2norm", &astra_ctvlib::matrix_2norm, "Calculate L2-Norm of Reconstruction");
    astra_ctvlib.def("data_distance", &astra_ctvlib::data_distance, "Calculate L2-Norm of Projection (aka Vectors)");
    astra_ctvlib.def("rmse", &astra_ctvlib::rmse, "Calculate reconstruction's RMSE");
    astra_ctvlib.def("original_tv", &astra_ctvlib::original_tv_3D, "Measure original TV");
    astra_ctvlib.def("tv_gd", &astra_ctvlib::tv_gd_3D, "3D TV Gradient Descent");
    astra_ctvlib.def("tv_fgp", &astra_ctvlib::tv_fgp_3D, "3D TV Fast Gradient Projection");
    astra_ctvlib.def("get_projections", &astra_ctvlib::get_projections, "Return the projection matrix to python");
    astra_ctvlib.def("get_model_projections", &astra_ctvlib::get_model_projections, "Return the re-projection matrix to python");
    astra_ctvlib.def("poisson_noise", &astra_ctvlib::poissonNoise, "Add Poisson Noise to Projections");
    astra_ctvlib.def("restart_recon", &astra_ctvlib::restart_recon, "Set all the Slices Equal to Zero");
}

// Save Reconstruction with Parallel MPI - I/O
// void astra_ctvlib::save_recon(char *filename) {
//     hid_t fd = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
//     hsize_t dims[3] = {Nslice, Ny, Nz};
//     hid_t dataspace = H5Screate_simple(3, dims, NULL);
//     hid_t dset = H5Dcreate(fd, "recon", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//     H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &recon.data[recon.index(0,0, 0)]);
//     H5Dclose(dset);
//     H5Sclose(dataspace);
//     H5Fclose(fd);
// }
