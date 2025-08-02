 //
//  astra_ctlib.cpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "mm_astra.hpp"
#include "container/matrix_ops.h"
#include "regularizers/tv_gd.h"
#include "regularizers/tv_fgp.h"

#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <iostream>
#include <cmath>
#include <random>
#include <vector>

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

mm_astra::mm_astra(int Ns, int Nray, int Nelements)
{
    //Intialize all the Member variables.
    Nslice = Ns;
    Ny = Nray;
    Nz = Nray;
    Nel = Nelements;

    //Initialize all the Slices in Recon as Zero.
    recon = Matrix4D(Nel,Ns,Ny,Nz); //Final Reconstruction.
}

mm_astra::mm_astra(int Ns, int Nray, int Nelements, Vec haadfAngles, Vec chemAngles)
{
    //Intialize all the Member variables.
    Nslice = Ns; Ny = Nray; Nz = Nray; nPix = Ny*Nz;
    Nel = Nelements; Ncol = Ny*Nz;
    NprojHaadf = haadfAngles.size();
    NprojChem = chemAngles.size();

    // Calculate number fo Rows in the Chemical and HAADF Measurement Matrices
    NrowChem = Nray*NprojChem; NrowHaadf = Nray*NprojHaadf;

    // Allocate the Chemical and HAADF Tilt Series at Runtime
    bh.resize(Nslice, NrowHaadf);
    bChem.resize(Nslice, NrowChem*Nel);
    gChem.resize(Nslice, NrowChem*Nel); 
    sigma.resize(Ns*Nray, Ns*Nray*Nel);
    
    //Initialize all the Slices in Recon as Zero.
    recon = Matrix4D(Nel,Ns,Ny,Nz); //Final Reconstruction.

    // Reshape Intermediate Variables for MM Reconstruction
    xx.resize(Nray*Nray*Nel); Ax.resize(NrowChem*Nel); 
    updateHAADF.resize(xx.size()); g.resize(Nslice,Nslice * NprojHaadf);
    updateCHEM.resize(xx.size());

    outProj.resize(NrowHaadf); outProj4D.resize(NrowChem*Nel);
    outVol.resize(Nray*Nray);  outVol4D.resize(xx.size());

    modelHAADF.resize(Nslice*Ny); 
    updateVol.resize(Nslice*Ny);

    // INITIALIZE ASTRA OBJECTS.

    // Create volume (2D) Geometry
    vol_geom = new CVolumeGeometry2D(Ny,Nz);
    
    // Create Volume ASTRA Object
    vol = new CFloat32VolumeData2D(vol_geom);
     
    // Specify projection matrix geometries
    float32 *hAng = new float32[NprojHaadf];
    float32 *cAng = new float32[NprojChem];

    for (int j = 0; j < NprojHaadf; j++) {
        hAng[j] = haadfAngles(j);    }

    for (int j = 0; j < NprojChem; j++){
        cAng[j] = chemAngles(j);     }
 
    // Create Projection Matrix Geometry
    haadfProjGeom = new CParallelProjectionGeometry2D(NprojHaadf,Nray,1.0,hAng);
    chemProjGeom = new CParallelProjectionGeometry2D(NprojChem,Nray,1.0,cAng);

    // Create Sinogram ASTRA Object
    haadfSino = new CFloat32ProjectionData2D(haadfProjGeom);
    chemSino = new CFloat32ProjectionData2D(chemProjGeom);

    // Create CUDA Projector
    hProj = new CCudaProjector2D(haadfProjGeom,vol_geom);
    cProj = new CCudaProjector2D(chemProjGeom,vol_geom);
}

// Set GPU ID (For Multi-GPU Reconstructions)
void mm_astra::set_gpu_id(int id){ 
    gpuID = id;
    recon.gpuIndex = id; }

// Return GPU ID (For Multi-GPU Reconstructions)
int mm_astra::get_gpu_id() { return gpuID; }

// Initialize Ground Truth
void mm_astra::initialize_initial_volume(){ 
    original_volume =  Matrix4D(Nel,Nslice,Ny,Nz); }

// Return if measuring HAADF Data Fusion 
bool mm_astra::get_measureHaadf(){ return measureHaadf; }

// Specify whether we want to measure HAADF Data Fusion 
void mm_astra::set_measureHaadf(bool inBool) { measureHaadf = inBool; }

// Return if measuring Poisson-Maximum Likelihood Data Fidelity 
bool mm_astra::get_measureChem(){  return measureChem; }

// Specify whether we want to measure Poisson-Maximum Likelihood Data Fidelity 
void mm_astra::set_measureChem(bool inBool) { measureChem = inBool; }

// Return Epsilon Offset 
float mm_astra::get_eps() { return eps; }

// Set Epsilon Offset
void mm_astra::set_eps(float inEps) { inEps = eps; }

// Return Gamma Parameter (HAADF term)
float mm_astra::get_gamma() { return gamma;}

// Set Gamma Parameter (HAADF term)
void mm_astra::set_gamma(float inGamma) { 
    gamma = inGamma;
    if (gamma != 1) { 
        spXXmatrix.resize( xx.size(),xx.size() ); 
        // Need to assign dummy values along diagonals
        for (int i=0; i < xx.size(); i++) { spXXmatrix.coeffRef(i,i) = 1; }}
    else { spXXmatrix.resize( 0,0 ); }
    cout << "Setting Gamma to: " << gamma << endl;
}

//Import tilt series (projections) from Python. 
void mm_astra::set_haadf_tilt_series(Mat in) { bh = in; }

//Import tilt series (projections) from Python. 
void mm_astra::set_chem_tilt_series(Mat in) {  bChem = in; }

// Set Reconstruction from Input 2D Slices
void mm_astra::set_recon(Mat inBuffer, int element, int slice) { recon.setData2D(inBuffer,element,slice); }

// Set Reconstruction from Input 2D Slices
void mm_astra::set_original_volume(Mat inBuffer, int element, int slice) { original_volume.setData2D(inBuffer,element,slice); }

// Initialize Forward Projection Operator
void mm_astra::initializeFP() { 
    algo_fp = new CCudaForwardProjectionAlgorithm();
    if (recon.gpuIndex != -1){algo_fp->setGPUIndex(recon.gpuIndex); } 
}

// Initialize Backprojection Operator
void mm_astra::initializeBP() { 
    algo_bp = new CCudaBackProjectionAlgorithm(); 
    if (recon.gpuIndex != -1){algo_bp->setGPUIndex(recon.gpuIndex); } 
}

// Foward project the data.
Vec mm_astra::forward_projection(const Vec &inVol)
{
    vol->copyData((float32*) &inVol(0));

    // Forward Project
    algo_fp->initialize(hProj,vol,haadfSino);
    if (recon.gpuIndex != -1){algo_fp->setGPUIndex(recon.gpuIndex); } 
    algo_fp->run();

    memcpy(&outProj(0), haadfSino->getData(), sizeof(float)*NrowHaadf);

    return outProj;
}

// Foward project the data - stack of chemical projections.
Vec mm_astra::forward_projection4D(const Vec &inVol)
{
    for (int e=0; e < Nel; e++) {
        // Return data from Astra
        vol->copyData((float32*) &inVol(e*Ny*Nz));

        // Forward Project
        algo_fp->initialize(cProj,vol,chemSino);
        if (recon.gpuIndex != -1){algo_fp->setGPUIndex(recon.gpuIndex); } 
        algo_fp->run();
        
        // Return data from Astra
        memcpy(&outProj4D(e*NrowChem), chemSino->getData(), sizeof(float)*NrowChem);
    }
    return outProj4D;
}

// Measure L2-norm for Chemical Tilt Series
float mm_astra::data_distance() {

    for (int s=0; s<Nslice; s++) {
        // Concatenate Elements ([e,s,:,:])
        for (int e=0; e<Nel; e++) {
            memcpy(&xx(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz); } 
        
        gChem.row(s) = forward_projection4D(xx);  }
    
    return (gChem - bChem).norm();
}

// Backproject the data (HAADF Volume).
Vec mm_astra::back_projection(const Vec &inProj) {   
    // Copy Data to Astra
    haadfSino->copyData((float32*) &inProj(0));

    // Forward Project
    algo_bp->initialize(hProj,haadfSino,vol);
    if (recon.gpuIndex != -1){algo_bp->setGPUIndex(recon.gpuIndex); } 
    algo_bp->run();

    // Return data from Astra
    memcpy(&outVol(0), vol->getData(), sizeof(float)*Ny*Nz);

    return outVol;
}

// Backproject the data - stack of volumes.
Vec mm_astra::back_projection4D(const Vec &inProj) {
    for (int e=0; e < Nel; e++) {
        // Copy Data to Astra
        chemSino->copyData((float32*) &inProj(e*NrowChem));

        // Back Project
        algo_bp->initialize(cProj,chemSino,vol);
        if (recon.gpuIndex != -1){algo_bp->setGPUIndex(recon.gpuIndex); } 
        algo_bp->run();
        
        // Return data from Astra
        memcpy(&outVol4D(e*Ny*Nz), vol->getData(), sizeof(float)*Ny*Nz);
    }
    return outVol4D;
}

// Estimate Lipschitz Parameters
void mm_astra::estimate_lipschitz() {
    VectorXf cc(xx.size()); cc.setOnes();
    L_Aps = (back_projection4D(forward_projection4D(cc))).maxCoeff();

    VectorXf hh(sigma.cols()); hh.setOnes();
    L_ASig = (sigma.transpose() * back_projection(forward_projection(sigma*hh))).maxCoeff(); 
}

// Pass Summation Matrix (sigma) to C++
void mm_astra::load_summation_matrix(Mat pySig) {
    // reset all the elements in sigma
    sigma.resize(Nslice*Ny, Nslice*Ny*Nel);
    for (int i=0; i <pySig.cols(); i++) {
        sigma.coeffRef(pySig(0,i), pySig(1,i)) = pySig(2,i);
    }
}

// Reconstruct with Poisson Maximum Likelihood
float mm_astra::poisson_ml(float lambdaCHEM) {
    float costCHEM = 0;

    for (int s=0; s< Nslice; s++){

        // Concatenate Elements ([e,s,:,:])
        for (int e=0; e<Nel; e++) {
            memcpy(&xx(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz); } 

        // (Poisson-ML)
        Ax = forward_projection4D(xx);
        updateCHEM = back_projection4D( (Ax - bChem.row(s).transpose()).array() / (Ax.array() + eps).array() );

        // Update along gradient direction
        xx -= lambdaCHEM/L_Aps * updateCHEM;

        // Return elements to recon.
        for (int e=0; e<Nel; e++) {
            memcpy(&recon.data[recon.index(e,s,0,0)], &xx(e*Ny*Nz), sizeof(float)*Ny*Nz); }

        // Measure Data Fidelity Cost  (Final Error)
        if (measureChem) { costCHEM += ( Ax.array() - bChem.row(s).transpose().array() * (Ax.array() + eps).log().array() ).sum();  }
    }
    // Apply Positivity
    recon.positivity();

    return costCHEM;
}

// Rescale bh with forward model
void mm_astra::rescale_projections() {
    for (int s=0; s<Nslice; s++){
        for (int e=0; e<Nel; e++) {
            memcpy(&xx(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz);
        }
        // Forward projection (HAADF)
        if (gamma == 1) { g.row(s) = forward_projection(sigma * xx); }
        else { g.row(s) = forward_projection( sigma * (Vec (xx.array().pow(gamma)) ) ); }
    }

    // Rescale projections
    for (int p=0; p<NprojHaadf; p++){ 
        // block (i,j,p,q) - block of size (p,q) starting at (i,j)
        bh.block(0,Ny*p,Nslice,Ny) /= bh.block(0,Ny*p,Nslice,Ny).maxCoeff();
        bh.block(0,Ny*p,Nslice,Ny) *=  g.block(0,Ny*p,Nslice,Ny).maxCoeff();
    }
}

// Initialize SIRT Reconstruction Operator
void mm_astra::initializeSIRT() {
    algo_sirt = new CCudaSirtAlgorithm();
    algo_sirt->initialize(hProj, haadfSino, vol);
    algo_sirt->setConstraints(true, 0, false, 1);
    if (recon.gpuIndex != -1){algo_sirt->setGPUIndex(recon.gpuIndex);}
}

// General SART Reconstruction Method
void mm_astra::SIRT(int e, int s, int nIter) {

    // Copy Data to Astra
    if (e >= 0) { 
        vol->copyData( (float32*) &recon.data[recon.index(e,s,0,0)] ); 
        chemSino->copyData( (float32*) &bChem(s,e*NrowChem) );      
        algo_sirt->updateSlice(chemSino, vol);                  
        
        // SIRT Reconstruction
        algo_sirt->run(nIter);

        // Return Data from Astra
        memcpy(&recon.data[recon.index(e,s,0,0)], vol->getData(), sizeof(float)*Ny*Nz); }                                                         
    else { 
        haadfSino->copyData( (float32*) &bh(s,0) );           
        vol->copyData( (float32*) &modelHAADF(0) );                   

        // SIRT Reconstruction
        algo_sirt->updateSlice(haadfSino, vol);  
        algo_sirt->run(nIter);        
        
        // Return Data from Astra
        memcpy(&updateVol(0), vol->getData(), sizeof(float)*Ny*Nz);     }                                                       
}

//  Non-Multi-Modal Reconstruction with SIRT
void mm_astra::chemical_SIRT(int nIter) { 
    // Vec chemProj(Ny*Ny); Vec outVol(Ny*Ny);
    for (int s=0; s<Nslice; s++) { 
        for (int e=0; e<Nel; e++) {
            SIRT(e,s,nIter);
        }
    }
}

// Initialize SART Reconstruction Operator
void mm_astra::initializeSART(std::string order) {
    projOrder = order;
    cout << "ProjectionOrder: " << projOrder << endl;
 
    algo_sart = new CCudaSartAlgorithm();
    algo_sart->initialize(hProj,haadfSino,vol);
    algo_sart->setConstraints(true, 0, false, 1);
    if (recon.gpuIndex != -1){algo_sart->setGPUIndex(recon.gpuIndex); }
}

// General SART Reconstruction Method
void mm_astra::SART(int e, int s, int nIter) {

    int Nproj = NrowHaadf / Ny;

    // Copy Data to Astra
    if (e >= 0) { 
        vol->copyData( (float32*) &recon.data[recon.index(e,s,0,0)] ); 
        chemSino->copyData( (float32*) &bChem(s,e*NrowChem) );      
        algo_sart->updateSlice(chemSino, vol);                  
        
        // SART Reconstruction
        algo_sart->run(Nproj * nIter); 

         // Return Data from Astra
        memcpy(&recon.data[recon.index(e,s,0,0)], vol->getData(), sizeof(float)*Ny*Nz);     }                                                         
    else { 

        haadfSino->copyData( (float32*) &bh(s,0) );
        vol->copyData( (float32*) &modelHAADF(0) );

        // SART Reconstruction
        algo_sart->updateSlice(haadfSino, vol);         
        algo_sart->run(Nproj * nIter);           
        
        // Return Data from Astra
        memcpy(&updateVol(0), vol->getData(), sizeof(float)*Ny*Nz);     }                                                       
}

//  Non-Multi-Modal Reconstruction with SIRT
void mm_astra::chemical_SART(int nIter) { 
    // Vec chemProj(Ny*Ny); Vec outVol(Ny*Ny);
    for (int s=0; s < Nslice; s++) { 
        for (int e=0; e<Nel; e++) {
            SART(e,s,nIter);
        }
    }
}

// SIRT Reconstruction.
Vec mm_astra::fuse(const Vec &inVol, int s, int nIter, std::string method) { 

    if (gamma == 1) {modelHAADF = sigma * inVol;}
    else            {modelHAADF = sigma * (Vec (inVol.array().pow(gamma)) ); }

    // Forward Project Weighted Chemistries against HAADF Tilt Series
    if (method ==  "SIRT") { SIRT(-1,s,nIter); }
    else                   { SART(-1,s,nIter); }

    // Back propagate the decomposition back to individual chemistries.  
    if (gamma == 1) {updateHAADF = sigma.transpose() * (updateVol - modelHAADF); }
    else {
        spXXmatrix.diagonal().array() = inVol.array().pow(gamma - 1); 
        updateHAADF = gamma * spXXmatrix * (Vec (sigma.transpose() * (updateVol - modelHAADF)) ); }
        
    return updateHAADF;
}

// Call Data Fusion with SIRT Projection Operator
tuple<float,float> mm_astra::call_sirt_data_fusion(float lambdaHAADF, float lambdaCHEM, int nIter)
{   return data_fusion(lambdaHAADF, lambdaCHEM, nIter, "SIRT"); }

// Call Data Fusion with SART Projection Operator
tuple<float,float> mm_astra::call_sart_data_fusion(float lambdaHAADF, float lambdaCHEM)
{   return data_fusion(lambdaHAADF, lambdaCHEM, 1, "SART"); }

// Data Fusion with SIRT Reconstruction on HAADF term
tuple<float,float> mm_astra::data_fusion(float lambdaHAADF, float lambdaCHEM, int nIter, std::string method) {
    float costHAADF = 0; float costCHEM = 0; 
    if (method == "SART" && (algo_sart == NULL)) { initializeSART("sequential"); }
    else if (method == "SIRT" && (algo_sirt == NULL)) { initializeSIRT(); }

    // Iterate Along slices
    for (int s=0; s < Nslice; s++) {

        // Iterate Along Elements
        for (int e=0; e< Nel; e++) {
            memcpy(&xx(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz); } 

        // // Compute HAADF Gradient Update
        if (gamma == 1) { g.row(s) = forward_projection( sigma * xx ); }
        else            { g.row(s) = forward_projection( sigma * (Vec (xx.array().pow(gamma)) ) ); }
        updateHAADF = fuse(xx,s,nIter,method);

        // (Poisson-ML)
        Ax = forward_projection4D(xx);
        updateCHEM = back_projection4D( (Ax - bChem.row(s).transpose()).array() / (Ax.array() + eps).array() );

        // Update along gradient directions
        xx -= lambdaCHEM/L_Aps * updateCHEM - lambdaHAADF * updateHAADF;

        // Iterate Along Elements
        for (int e=0; e< Nel; e++) {
            memcpy(&recon.data[recon.index(e,s,0,0)], &xx(e*Ny*Nz), sizeof(float)*Ny*Nz); } 

        // Measure Data Fidelity Cost  
        if (measureChem) {costCHEM += ( Ax.array() - bChem.row(s).transpose().array() * (Ax.array() + eps).log().array() ).sum(); }
    }

    // Apply Positivity
    recon.positivity();

    // Measure Multi-Modal (HAADF) Cost 
    if (measureHaadf) costHAADF = (g - bh).norm();

    return make_tuple(costHAADF,costCHEM);
}

// TV Minimization (Gradient Descent)
float mm_astra::tv_gd_4D(int ng, float lambdaTV) { return cuda_tv_gd_4D(recon.data, ng, lambdaTV, Nslice, Ny, Nz, Nel, gpuID); }

// TV Minimization (Fast Gradient Projection Method)
float mm_astra::tv_fgp_4D(int ng, float lambdaTV) { return cuda_tv_fgp_4D(recon.data, ng, lambdaTV, Nslice, Ny, Nz, Nel, gpuID); }

// TV Minimization (Split Bregman)
// float mm_astra::tv_sb_4D(int ng, float lambdaTV) { return cuda_tv_sb_4D(recon.data, ng, lambdaTV, Nslice, Ny, Nz, Nel, gpuID);}

// Measure the RMSE 
Vec mm_astra::rmse() { 
    // Rescale Recon
    return Map<Vec>(cuda_rmse_4D(recon.data, original_volume.data, Nslice, Ny, Nz, Nel),Nel);  }

// Return Reconstruction to Python.
Mat mm_astra::get_recon(int element, int slice) { return recon.getData2D(element, slice); }
Mat mm_astra::get_gt(int element, int slice) { return original_volume.getData2D(element, slice); }

//Return the projections.
Mat mm_astra::get_model_projections() { return g; }
Mat mm_astra::get_haadf_projections() { return bh; }
Mat mm_astra::get_chem_projections()  { return bChem; }

// Restart the Reconstruction (Reset Values to Zero). 
void mm_astra::restart_recon() { memset(recon.data, 0, sizeof(float)*Nslice*Ny*Nz*Nel); }

//Python functions for mm_astra module.
PYBIND11_MODULE(mm_astra, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions using ASTRA Cuda Library";
    py::class_<mm_astra> mm_astra(m, "mm_astra");
    mm_astra.def(py::init<int,int,int>());
    mm_astra.def(py::init<int,int,int,Vec,Vec>());
    mm_astra.def("initialize_initial_volume", &mm_astra::initialize_initial_volume, "Initalize Original Data (Ground Truth)");
    mm_astra.def("set_haadf_tilt_series", &mm_astra::set_haadf_tilt_series, "Pass the Projections to C++ Object");
    mm_astra.def("set_chem_tilt_series", &mm_astra::set_chem_tilt_series, "Pass the Projections to C++ Object");
    mm_astra.def("set_measureHaadf", &mm_astra::set_measureHaadf, "Flag to Measure Haadf Data Fusion Term");
    mm_astra.def("set_measureChem", &mm_astra::set_measureChem, "Flag to Measure Poisson-ML Data Fidelity Term");
    mm_astra.def("set_gamma", &mm_astra::set_gamma, "Set Gamma");
    mm_astra.def("set_recon", &mm_astra::set_recon, "Pass the Reconstruction from Python to C++");
    mm_astra.def("get_gpu_id", &mm_astra::get_gpu_id, "Get the GPU ID");
    mm_astra.def("set_gpu", &mm_astra::set_gpu_id, "Set the GPU ID");
    mm_astra.def("set_original_volume", &mm_astra::set_original_volume, "Pass the Ground Truth to C++ Object");
    mm_astra.def("measureHaadf", &mm_astra::get_measureHaadf, "Return measureHaadf");
    mm_astra.def("measureChem", &mm_astra::get_measureChem, "Return measureHaadf");
    mm_astra.def("gamma", &mm_astra::get_gamma, "Get Gamma");
    mm_astra.def("data_distance", &mm_astra::data_distance, "Data Distance");
    mm_astra.def("load_sigma", &mm_astra::load_summation_matrix, "Load Summation Matrix (Sigma)");
    mm_astra.def("initialize_FP", &mm_astra::initializeFP, "Initialize Forward Projection");
    mm_astra.def("initialize_BP", &mm_astra::initializeBP, "Initialize Back Projection");
    mm_astra.def("forward_projection", &mm_astra::forward_projection, "Forward Projection");
    mm_astra.def("estimate_lipschitz", &mm_astra::estimate_lipschitz, "Estimate Lispchitz Parameters");
    mm_astra.def("rescale_projections", &mm_astra::rescale_projections, "Rescale Experimental HAADF Projections");
    mm_astra.def("poisson_ml", &mm_astra::poisson_ml, "Reconstruct Data from Poisson Maximum Likelihood");
    mm_astra.def("data_fusion", &mm_astra::data_fusion, "Data Fusion");
    mm_astra.def("tv_gd", &mm_astra::tv_gd_4D, "3D TV Gradient Descent");
    mm_astra.def("tv_fgp_4D", &mm_astra::tv_fgp_4D, "3D TV Fast Gradient Projection");
    mm_astra.def("initialize_SART", &mm_astra::initializeSART, "Initialize SART");
    mm_astra.def("chemical_SART", &mm_astra::chemical_SART, "SART on the Raw Chemical Projections");
    mm_astra.def("initialize_SIRT", &mm_astra::initializeSIRT, "Initialize SIRT");
    mm_astra.def("chemical_SIRT", &mm_astra::chemical_SIRT, "SIRT on the Raw Chemical Projections");
    mm_astra.def("sirt_data_fusion", &mm_astra::call_sirt_data_fusion, "Data Fusion with SIRT Forward Projection");
    mm_astra.def("sart_data_fusion", &mm_astra::call_sart_data_fusion, "Data Fusion with SIRT Forward Projection");    
    mm_astra.def("rmse", &mm_astra::rmse, "Measure RMSE");
    mm_astra.def("get_recon", &mm_astra::get_recon, "Return the Reconstruction to Python");
    mm_astra.def("get_haadf_projections", &mm_astra::get_haadf_projections, "Return the projection matrix to python");
    mm_astra.def("get_chem_projections", &mm_astra::get_chem_projections, "Return the projection matrix to python");
    mm_astra.def("get_model_projections", &mm_astra::get_model_projections, "Return the re-projection matrix to python");
    mm_astra.def("get_gpu", &mm_astra::get_gpu_id, "Get GPU ID");
    mm_astra.def("restart_recon", &mm_astra::restart_recon, "Set all the Slices Equal to Zero");
}

// mm_astra.def("SIRT_data_fusion", &mm_astra::SIRT_data_fusion, "Data Fusion with SIRT Algorithm");
// mm_astra.def("SART_data_fusion", &mm_astra::SART_data_fusion, "Data Fusion with SART Algorithm");

// mm_astra.def("SART", &mm_astra::SART, "ART Reconstruction");
// mm_astra.def("SIRT", &mm_astra::SIRT, "SIRT Reconstruction");
// mm_astra.def("tv_3D", &mm_astra::tv_3D, "Measure TV of the reconstruction");
// mm_astra.def("print_recon", &mm_astra::print_recon, "Print the reconstruction variable - debug");
// mm_astra.def("save_recon", &mm_astra::save_recon, "Save the Reconstruction with HDF5");

// void mm_astra::print_recon(){
//     printf((recon.data).size());
// }

// //Measure Reconstruction's TV.
// float mm_astra::tv_3D(int NelSel) { 
    
//     // for (int s=0; s < Nslice; s++) {
//     //     recon_small[recon_small.index(s,0,0)] = recon.data[recon.index(NelSel,s,0,0)];
//     // }
//     return (cuda_tv_3D(recon.data[recon.index()], Nslice, Ny, Nz));
// }

// // Copy Data to Astra
// chemSino->copyData((float32*) &inProj(e*NrowChem));

// // Back Project
// algo_bp->initialize(cProj,chemSino,vol);
// algo_bp->run();

// // Return data from Astra
// memcpy(&outVol4D(e*Ny*Nz), vol->getData(), sizeof(float)*Ny*Nz);

// Data Fusion with SIRT Reconstruction on HAADF term
// tuple<float,float> mm_astra::SART_data_fusion(float lambdaHAADF, float lambdaCHEM, int nIter)
// {
//     float costHAADF = 0; float costCHEM = 0; 

//     // Iterate Along slices
//     for (int s=0; s < Nslice; s++) {

//         // Iterate Along Elements
//         for (int e=0; e< Nel; e++) {
//             memcpy(&xx(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz); } 

//         // Forward projection (HAADF)
//         if (gamma == 1) { g.row(s) = forward_projection(sigma * xx); }
//         else { g.row(s) = forward_projection( sigma * (Vec (xx.array().pow(gamma)) ) ); }

//         // // Compute SART Gradient Update
//         // updateHAADF = SART(xx,s,nIter);

//         // xx -= lambdaHAADF * updateHAADF;

//         // (Poisson-ML)
//         Ax = forward_projection4D(xx);
//         updateCHEM = back_projection4D( (Ax - bChem.row(s).transpose()).array() / (Ax.array() + eps).array() );

//         // New Version?
//         xx -= lambdaCHEM/L_Aps * updateCHEM;
//         updateHAADF = SART(xx,s,nIter);
//         xx -= lambdaHAADF * updateHAADF;

//         // Update along gradient directions
//         // xx -= lambdaCHEM/L_Aps * updateCHEM - lambdaHAADF * updateHAADF;

//         // Iterate Along Elements
//         for (int e=0; e< Nel; e++) {
//             memcpy(&recon.data[recon.index(e,s,0,0)], &xx(e*Ny*Nz), sizeof(float)*Ny*Nz); } 

//         // Measure Data Fidelity Cost  
//         if (measureChem) {costCHEM += ( Ax.array() - bChem.row(s).transpose().array() * (Ax.array() + eps).log().array() ).sum(); }
//     }

//     // Apply Positivity
//     recon.positivity();

//     // rescale_projections();

//     // Measure Multi-Modal (HAADF) Cost 
//     if (measureHaadf) costHAADF = (g - bh).norm();

//     return make_tuple(costHAADF,costCHEM);
// }

// Data Fusion with SART Reconstruction.
// Vec mm_astra::SART_fusion(const Vec &inVol, int s, int nIter) { 
//     Vec tmpHAADF(sigma.rows()); Vec updateSART(sigma.rows());

//     int Nproj = NrowHaadf / Ny;

//     if (gamma == 1) {tmpHAADF = sigma * inVol;}
//     else            {tmpHAADF = sigma * (Vec (inVol.array().pow(gamma)) ); }

//     // Pass 2D Slice and Sinogram (Measurements) to ASTRA
//     haadfSino->copyData((float32*) &bh(s,0));
//     vol->copyData( (float32*) &tmpHAADF(0) );

//     // SIRT Reconstruction
//     algo_sart->updateSlice(haadfSino, vol);
//     algo_sart->run(nIter * Nproj);

//     // Return Slice to tomo_TV
//     memcpy(&updateSART(0), vol->getData(), sizeof(float)*Ny*Nz);
//     if (gamma == 1) {updateHAADF = sigma.transpose() * (updateSART - tmpHAADF); }
//     else {
//         // # pragma omp parallel for
//         // for (int i=0; i < inVol.size(); i++) { spXXmatrix.coeffRef(i,i) = pow(inVol(i),gamma-1); }
//         spXXmatrix.diagonal().array() = inVol.array().pow(gamma - 1); 
//         updateHAADF = gamma * spXXmatrix * (Vec (sigma.transpose() * (updateSART - tmpHAADF)) ); }
        
//     return updateHAADF;

// }

// // Fused Multi-Modal Tomography Reconstruction
// tuple<float,float> mm_astra::data_fusion(float lambdaHAADF, float lambdaCHEM) {
//     float costHAADF = 0; float costCHEM = 0; 

//     for (int s=0; s< Nslice; s++){

//         // Concatenate Elements ([e,s,:,:])
//         for (int e=0; e<Nel; e++) {
//             memcpy(&xx(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz); } 

//         // Forward projection (HAADF)
//         if (gamma == 1) { 
//             g.row(s) = forward_projection(sigma * xx); 
//             updateHAADF = sigma.transpose() * back_projection( Vec (g.row(s) - bh.row(s)) ); }
//         else {
//             g.row(s) = forward_projection( sigma * (Vec (xx.array().pow(gamma)) ) ); 
//             spXXmatrix.diagonal().array() = xx.array().pow(gamma - 1); 
//             updateHAADF = gamma * spXXmatrix * sigma.transpose() * back_projection( Vec (g.row(s) - bh.row(s)) ); }
        
//         // (Poisson-ML)
//         Ax = forward_projection4D(xx);
//         updateCHEM = back_projection4D( (Ax - bChem.row(s).transpose()).array() / (Ax.array() + eps).array() );
        
//         // Update along gradient directions
//         xx -= lambdaHAADF/L_ASig * updateHAADF + lambdaCHEM/L_Aps * updateCHEM;
        
//         // Return elements to recon.
//         for (int e=0; e<Nel; e++) {
//             memcpy(&recon.data[recon.index(e,s,0,0)], &xx(e*Ny*Nz), sizeof(float)*Ny*Nz); }

//         // Measure Data Fidelity Cost  
//         if (measureChem) {costCHEM += ( Ax.array() - bChem.row(s).transpose().array() * (Ax.array() + eps).log().array() ).sum(); }
//     }
//     // Measure Multi-Modal (HAADF) Cost 
//     if (measureHaadf) costHAADF = (g - bh).norm();

//     return make_tuple(costHAADF,costCHEM);
// }