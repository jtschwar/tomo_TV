//
//  astra_ctlib.cpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "astra_ctvlib.hpp"
#include "regularizers/tv_gd.h"
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

astra_ctvlib::astra_ctvlib(int Ns, int Nray, int Nproj, Vec pyAngles)
{
    //Intialize all the Member variables.
    Nslice = Ns;
    Ny = Nray;
    Nz = Nray;
    Nrow = Nray*Nproj;
    Ncol = Ny*Nz;
    b.resize(Nslice, Nrow);
    g.resize(Nslice, Nrow);
    
    //Initialize all the Slices in Recon as Zero.
    recon = Matrix3D(Ns,Ny,Nz); //Final Reconstruction.
    temp_recon =  Matrix3D(Ns,Ny,Nz); // Temporary copy for measuring changes in TV and ART.

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
}

void astra_ctvlib::initilizeInitialVolume()
{
    original_volume = Matrix3D(Nslice,Ny,Nz);
}

void astra_ctvlib::checkNumGPUs()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);

    cout << "NumGPUs: " << nDevices << endl;
}
//Import tilt series (projections) from Python.
void astra_ctvlib::setTiltSeries(Mat in)
{
    b = in;
}

// Import the original volume from python.
void astra_ctvlib::setOriginalVolume(Mat inBuffer, int slice)
{
    original_volume.setData(inBuffer,slice);
}

void astra_ctvlib::setRecon(Mat inBuffer, int slice)
{
    recon.setData(inBuffer,slice);
}

// Create projections from Volume (for simulation studies)
void astra_ctvlib::create_projections()
{
     // Create Sinogram ASTRA Object
     sino = new CFloat32ProjectionData2D(proj_geom);

     // Create CUDA Projector
     proj = new CCudaProjector2D(proj_geom,vol_geom);
     
     // Forward Projection Operator
     algo_fp = new CCudaForwardProjectionAlgorithm();

    int sliceInd;
    for (int s=0; s < Nslice; s++) {
        
        // Pass Input Volume to Astra
        sliceInd = original_volume.calc_index(s,0,0);
        vol->copyData( (float32*) &original_volume.data[sliceInd] );

        // Forward Project
        algo_fp->initialize(proj,vol,sino);
        algo_fp->run();

        // Return Sinogram (Measurements) to tomoTV
        memcpy(&b(s,0), sino->getData(), sizeof(float)*Nrow);
    }
}

// Add poisson noise to projections.
void astra_ctvlib::poissonNoise(int Nc)
{
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

void astra_ctvlib::update_projection_angles(int Nproj, Vec pyAngles)
{
    // Specify projection matrix geometries
    float32 *angles = new float32[pyAngles.size()];

    for (int j = 0; j < pyAngles.size(); j++) {
        angles[j] = pyAngles(j);    }
    
    // Create Projection Matrix Geometry and Projector.
    proj_geom = new CParallelProjectionGeometry2D(Nproj, Ny, 1, angles);
    proj = new CCudaProjector2D(proj_geom,vol_geom);
    sino =  new CFloat32ProjectionData2D(proj_geom);
}

void astra_ctvlib::initializeSART(std::string order)
{
    projOrder = order;
    cout << "ProjectionOrder: " << projOrder << endl;
 
    algo_sart = new CCudaSartAlgorithm();
    algo_sart->initialize(proj,sino,vol);
    algo_sart->setConstraints(true, 0, false, 1);
}

// ART Reconstruction.
void astra_ctvlib::SART(float beta)
{
    int Nproj = Nrow / Ny;
    algo_sart->updateProjOrder(projOrder);
    
    int sliceInd;
    for (int s=0; s < Nslice; s++) {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA.
        sliceInd = recon.calc_index(s,0,0);
        sino->copyData((float32*) &b(s,0));
        vol->copyData( (float32*) &recon.data[sliceInd] );

        // SART Reconstruction
        algo_sart->updateSlice(sino,vol);
    
        algo_sart->run(Nproj);
        
        // Return Slice to tomo_TV
        memcpy(&recon.data[sliceInd], vol->getData(), sizeof(float)*Ny*Nz);
    }
}

void astra_ctvlib::initializeSIRT()
{
    algo_sirt = new CCudaSirtAlgorithm();
   // algo_sirt->setConstraints(true, 0, false, 1);
}

// SIRT Reconstruction.
void astra_ctvlib::SIRT(float beta)
{
    int sliceInd;
    for (int s=0; s < Nslice; s++)
    {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA.
        sliceInd = recon.calc_index(s,0,0);
        sino->copyData((float32*) &b(s,0));
        vol->copyData( (float32*) &recon.data[sliceInd] );

        // SIRT Reconstruction
        algo_sirt->initialize(proj,sino,vol);
        algo_sirt->setConstraints(true, 0, false, 1);
        algo_sirt->run(1);
        
        // Return Slice to tomo_TV
        memcpy(&recon.data[sliceInd], vol->getData(), sizeof(float)*Ny*Nz);
    }
}

void astra_ctvlib::initializeFBP(std::string filter)
{
    // Possible Inputs for FilterType:
    // none, ram-lak, shepp-logan, cosine, hamming, hann, tukey, lanczos,
    // triangular, gaussian, barlett-hann, blackman, nuttall, blackman-harris,
    // blackman-nuttall, flat-top, kaiser, parzen
    
   fbfFilter = filter;
   cout << "FBP Filter: " << filter << endl;
   algo_fbp = new CCudaFilteredBackProjectionAlgorithm();
}

// Filtered Backprojection.
void astra_ctvlib::FBP()
{
    E_FBPFILTER fbfFilt = convertStringToFilter(fbfFilter);
    for (int s=0; s < Nslice; s++)
    {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA.
        sino->copyData((float32*) &b(s,0));
        vol->copyData((float32*) &recon.data[recon.calc_index(s,0,0)]);

        // SIRT Reconstruction
        algo_fbp->initialize(sino,vol,fbfFilt);
       
        algo_fbp->run();
        
        // Return Slice to tomo_TV
        memcpy(&recon.data[recon.calc_index(s,0,0)], vol->getData(), sizeof(float)*Ny*Nz);
    }
}

// Create Local Copy of Reconstruction. 
void astra_ctvlib::copy_recon()
{
    memcpy(temp_recon.data, recon.data, sizeof(float)*Nslice*Ny*Nz);
}

// Measure the 2 norm between temporary and current reconstruction.
float astra_ctvlib::matrix_2norm()
{
    float L2;
    L2 = cuda_euclidean_dist(recon.data, temp_recon.data, Nslice, Ny, Nz);
    return L2;
}

// Measure the 2 norm between experimental and reconstructed projections.
float astra_ctvlib::vector_2norm()
{
  return (g - b).norm() / g.size(); // Nrow*Nslice,sum_{ij} M_ij^2 / Nrow*Nslice
}

// Measure the 2 norm for projections when data is 'dynamically' collected.
float astra_ctvlib::dyn_vector_2norm(int dyn_ind)
{
    dyn_ind *= Ny;
    return ( g.leftCols(dyn_ind) - b.leftCols(dyn_ind) ).norm() / g.leftCols(dyn_ind).size();
}

// Foward project the data.
void astra_ctvlib::forwardProjection()
{
    for (int s=0; s < Nslice; s++) {
        
        vol->copyData((float32*) &recon.data[recon.calc_index(s,0,0)]);

        // Forward Project
        algo_fp->initialize(proj,vol,sino);
        algo_fp->run();

        memcpy(&g(s,0), sino->getData(), sizeof(float)*Nrow);
    }
}

// Measure the RMSE (simulation studies)
float astra_ctvlib::rmse()
{
    float rmse;
    rmse = cuda_rmse(recon.data, original_volume.data, Nslice, Ny, Nz);
    return rmse;
}

//Measure Original Volume's TV.
float astra_ctvlib::original_tv_3D()
{
    float tv;
    tv = cuda_tv_3D(original_volume.data, Nslice, Ny, Nz);
    return tv;
}

// TV Minimization (Gradient Descent)
float astra_ctvlib::tv_gd_3D(int ng, float dPOCS)
{
    float tv;
    tv = cuda_tv_gd_3D(recon.data, ng, dPOCS, Nslice, Ny, Nz);
    return tv;
}

// Return Reconstruction to Python.
Mat astra_ctvlib::getRecon(int slice)
{
    return recon.getData(slice);
}

//Return the projections.
Mat astra_ctvlib::get_projections()
{
    return b;
}

// Restart the Reconstruction (Reset to Zero). 
void astra_ctvlib::restart_recon()
{
    memset(recon.data, 0, sizeof(float)*Nslice*Ny*Nz);
}

//Python functions for astra_ctvlib module.
PYBIND11_MODULE(astra_ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions using ASTRA Cuda Library";
    py::class_<astra_ctvlib> astra_ctvlib(m, "astra_ctvlib");
    astra_ctvlib.def(py::init<int,int,int,Vec>());
    astra_ctvlib.def("initilizeInitialVolume", &astra_ctvlib::initilizeInitialVolume, "Initialize Original Data");
    astra_ctvlib.def("setTiltSeries", &astra_ctvlib::setTiltSeries, "Pass the Projections to C++ Object");
    astra_ctvlib.def("setOriginalVolume", &astra_ctvlib::setOriginalVolume, "Pass the Volume to C++ Object");
    astra_ctvlib.def("create_projections", &astra_ctvlib::create_projections, "Create Projections from Volume");
    astra_ctvlib.def("getRecon", &astra_ctvlib::getRecon, "Return the Reconstruction to Python");
    astra_ctvlib.def("setRecon", &astra_ctvlib::setRecon, "Return the Reconstruction to Python");
    astra_ctvlib.def("initializeSART", &astra_ctvlib::initializeSART, "Generate Config File");
    astra_ctvlib.def("SART", &astra_ctvlib::SART, "ART Reconstruction");
    astra_ctvlib.def("initializeSIRT", &astra_ctvlib::initializeSIRT, "Generate Config File");
    astra_ctvlib.def("SIRT", &astra_ctvlib::SIRT, "SIRT Reconstruction");
    astra_ctvlib.def("initializeFBP", &astra_ctvlib::initializeFBP, "Generate Config File");
    astra_ctvlib.def("FBP", &astra_ctvlib::FBP, "Filtered Backprojection");
    astra_ctvlib.def("forwardProjection", &astra_ctvlib::forwardProjection, "Forward Projection");
    astra_ctvlib.def("copy_recon", &astra_ctvlib::copy_recon, "Copy the reconstruction");
    astra_ctvlib.def("matrix_2norm", &astra_ctvlib::matrix_2norm, "Calculate L2-Norm of Reconstruction");
    astra_ctvlib.def("vector_2norm", &astra_ctvlib::vector_2norm, "Calculate L2-Norm of Projection (aka Vectors)");
    astra_ctvlib.def("dyn_vector_2norm", &astra_ctvlib::dyn_vector_2norm, "Calculate L2-Norm of Partially Sampled Projections (aka Vectors)");
    astra_ctvlib.def("rmse", &astra_ctvlib::rmse, "Calculate reconstruction's RMSE");
    astra_ctvlib.def("original_tv", &astra_ctvlib::original_tv_3D, "Measure original TV");
    astra_ctvlib.def("tv_gd", &astra_ctvlib::tv_gd_3D, "3D TV Gradient Descent");
    astra_ctvlib.def("get_projections", &astra_ctvlib::get_projections, "Return the projection matrix to python");
    astra_ctvlib.def("poissonNoise", &astra_ctvlib::poissonNoise, "Add Poisson Noise to Projections");
    astra_ctvlib.def("restart_recon", &astra_ctvlib::restart_recon, "Set all the Slices Equal to Zero");
    astra_ctvlib.def("gpuCount", &astra_ctvlib::checkNumGPUs, "Check Num GPUs available");
}
