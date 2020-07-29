//
//  astra_ctlib.cpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "astra_ctvlib.hpp"

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
    b.resize(Ny, Nrow);
    g.resize(Ny, Nrow);
    
    //Initialize all the Slices in Recon as Zero.
    recon = new Mat[Nslice]; //Final Reconstruction.
    temp_recon = new Mat[Nslice]; // Temporary copy for measuring changes in TV and ART.
    tv_recon = new Mat[Nslice]; // Temporary copy for measuring 3D TV - Derivative. 
    original_volume = new Mat[Nslice]; // Original Volume for Simulation Studies.
    
    // Initialize the 3D Matrices.
    for (int i=0; i < Nslice; i++)
    {
        recon[i] = Mat::Zero(Ny, Nz);
        temp_recon[i] = Mat::Zero(Ny, Nz);
        tv_recon[i] = Mat::Zero(Ny,Nz);
    }
    
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
void astra_ctvlib::setOriginalVolume(Mat in, int slice)
{
    original_volume[slice] = in;
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

    for (int s=0; s < Nslice; s++) {
        
        // Pass Input Volume to Astra
        vol->copyData((float32*) &original_volume[s](0,0));

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
    for (int s=0; s < Nslice; s++) {
       
       // Pass 2D Slice and Sinogram (Measurements) to ASTRA.
       sino->copyData((float32*) &b(s,0));
       vol->copyData((float32*) &recon[s](0,0));

        // SART Reconstruction
       algo_sart->updateSlice(sino,vol);
    
       // Add Positivity Constraint, Second Input Pair is for Max Constraint. 
       //algo_sart->setConstraints(true, 0, false, 1);
       algo_sart->run(Nproj);
        
       // Return Slice to tomo_TV
       memcpy(&recon[s](0,0), vol->getData(), sizeof(float)*Ny*Nz);
    }
}

//Calculate Lipshits Gradient (for SIRT).
float astra_ctvlib::lipschits()
{
    // TODO: Find Lipschitz Constant with ASTRA Projector.
//    VectorXf f(Ncol), L(Ncol);
//    f.setOnes();
    
    
}

void astra_ctvlib::initializeSIRT()
{
   algo_sirt = new CCudaSirtAlgorithm();
}

// SIRT Reconstruction.
void astra_ctvlib::SIRT(float beta)
{
    for (int s=0; s < Nslice; s++)
    {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA.
        sino->copyData((float32*) &b(s,0));
        vol->copyData((float32*) &recon[s](0,0));

        // SIRT Reconstruction
        algo_sirt->initialize(proj,sino,vol);
       
        // Add Positivity Constraint, Second Input Pair is for Max Constraint. 
        algo_sirt->setConstraints(true, 0, false, 1);
        algo_sirt->run(1);
        
        // Return Slice to tomo_TV
        memcpy(&recon[s](0,0), vol->getData(), sizeof(float)*Ny*Nz);
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
        vol->copyData((float32*) &recon[s](0,0));

        // SIRT Reconstruction
        algo_fbp->initialize(sino,vol,fbfFilt);
       
        algo_fbp->run();
        
        // Return Slice to tomo_TV
        memcpy(&recon[s](0,0), vol->getData(), sizeof(float)*Ny*Nz);
    }
}

// Remove Negative Voxels.
void astra_ctvlib::positivity()
{
    #pragma omp parallel for
    for(int i=0; i<Nslice; i++)
    {
        recon[i] = (recon[i].array() < 0).select(0, recon[i]);
    }
}

// Create Local Copy of Reconstruction. 
void astra_ctvlib::copy_recon()
{
    memcpy(temp_recon, recon, sizeof(recon));
}

// Measure the 2 norm between temporary and current reconstruction.
float astra_ctvlib::matrix_2norm()
{
    float L2;
    #pragma omp parallel for reduction(+:L2)
    for (int s =0; s < Nslice; s++)
    {
        L2 += ( recon[s].array() - temp_recon[s].array() ).square().sum();
    }
    return sqrt(L2);
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
        
        vol->copyData((float32*) &recon[s](0,0));

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
    #pragma omp parallel for reduction(+:rmse)
    for (int s = 0; s < Nslice; s++)
    {
        rmse += ( recon[s].array() - original_volume[s].array() ).square().sum();
    }
    rmse = sqrt( rmse / (Nslice * Ny * Nz ) );
    return rmse;
}

//Measure Reconstruction's TV.
float astra_ctvlib::tv_3D()
{
    float tv;
    float eps = 1e-8;
    int nx = Nslice;
    int ny = Ny;
    int nz = Nz;
    
    #pragma omp parallel for
    for (int i = 0; i < Nslice; i++)
    {
        int ip = (i+1)%nx;
        for (int j = 0; j < ny; j++)
        {
            int jp = (j+1)%ny;
            for (int k = 0; k < nz; k++)
            {
                int kp = (k+1)%ny;
                tv_recon[i](j,k) = sqrt(eps + ( recon[i](j,k) - recon[ip](j,k) ) * ( recon[i](j,k) - recon[ip](j,k) )
                                        + ( recon[i](j,k) - recon[i](jp,k) ) * ( recon[i](j,k) - recon[i](jp,k) )
                                        + ( recon[i](j,k) - recon[i](j,kp) ) * ( recon[i](j,k) - recon[i](j,kp) ));
            }
        }
    }

    #pragma omp parallel for reduction(+:tv)
    for (int i = 0; i < Nslice; i++)
    {
        tv += tv_recon[i].sum();
    }
    return tv;
}

//Measure Original Volume's TV.
float astra_ctvlib::original_tv_3D()
{
    float tv;
    float eps = 1e-8;
    int nx = Nslice;
    int ny = Ny;
    int nz = Nz;
    
    #pragma omp parallel for
    for (int i = 0; i < Nslice; i++)
    {
        int ip = (i+1)%nx;
        for (int j = 0; j < ny; j++)
        {
            int jp = (j+1)%ny;
            for (int k = 0; k < nz; k++)
            {
                int kp = (k+1)%ny;
                tv_recon[i](j,k) = sqrt(eps + pow( original_volume[i](j,k) - original_volume[ip](j,k) , 2)
                                        + pow( original_volume[i](j,k) - original_volume[i](jp,k) , 2)
                                        + pow( original_volume[i](j,k) - original_volume[i](j,kp) , 2));
            }
        }
    }
    
    #pragma omp parallel for reduction(+:tv)
    for (int i = 0; i < Nslice; i++)
    {
        tv += tv_recon[i].sum();
    }
    return tv;
}

// TV Minimization (Gradient Descent)
void astra_ctvlib::tv_gd_3D(int ng, float dPOCS)
{
    float eps = 1e-8;
    float tv_norm;
    int nx = Nslice;
    int ny = Ny;
    int nz = Nz;
    
    //Calculate TV Derivative Tensor.
    for(int g=0; g < ng; g++)
    {
        #pragma omp parallel for reduction(+:tv_norm)
        for (int i = 0; i < Nslice; i++)
        {
            int ip = (i+1) % nx;
            int im = (i-1+nx) % nx;
            for (int j = 0; j < ny; j++)
            {
                int jp = (j+1) % ny;
                int jm = (j-1+ny) % ny;
                for (int k = 0; k < ny; k++)
                {
                    int kp = (k+1)%ny;
                    int km = (k-1+ny)%ny;
                    
                    float v1n = 3.0*recon[i](j, k) - recon[ip](j, k) - recon[i](jp, k) - recon[i](j, kp);
                    float v1d = sqrt(eps + ( recon[i](j, k) - recon[ip](j, k) ) * ( recon[i](j, k) - recon[ip](j, k) )
                                      +  ( recon[i](j, k) - recon[i](jp, k) ) * ( recon[i](j, k) - recon[i](jp, k) )
                                      +  ( recon[i](j, k) - recon[i](j, kp) ) * ( recon[i](j, k) - recon[i](j, kp) ));
                    float v2n = recon[i](j, k) - recon[im](j, k);
                    float v2d = sqrt(eps + ( recon[im](j, k) - recon[i](j, k) ) * ( recon[im](j, k) - recon[i](j, k) )
                                      +  ( recon[im](j, k) - recon[im](jp, k) ) * ( recon[im](j, k) - recon[im](jp, k) )
                                      +  ( recon[im](j, k) - recon[im](j, kp)) * ( recon[im](j, k) - recon[im](j, kp)));
                    float v3n = recon[i](j, k) - recon[i](jm, k);
                    float v3d = sqrt(eps + ( recon[i](jm, k) - recon[ip](jm, k) ) * ( recon[i](jm, k) - recon[ip](jm, k) )
                                      +  ( recon[i](jm, k) - recon[i](j, k) ) * ( recon[i](jm, k) - recon[i](j, k) )
                                      +  ( recon[i](jm, k) - recon[i](jm, kp) ) * ( recon[i](jm, k) - recon[i](jm, kp) ) );
                    float v4n = recon[i](j, k) - recon[i](j, km);
                    float v4d = sqrt(eps + ( recon[i](j, km) - recon[ip](j, km)) * ( recon[i](j, km) - recon[ip](j, km))
                                      + ( recon[i](j, km) - recon[i](jp, km)) * ( recon[i](j, km) - recon[i](jp, km))
                                      + ( recon[i](j, km) - recon[i](j, k) ) * ( recon[i](j, km) - recon[i](j, k) ) );
                    tv_recon[i](j,k) = v1n/v1d + v2n/v2d + v3n/v3d + v4n/v4d;
                    tv_norm += tv_recon[i](j,k) * tv_recon[i](j,k);
                }
            }
        }
        tv_norm = sqrt(tv_norm);
        
        // Gradient Descent.
        #pragma omp parallel for
        for (int l = 0; l < Nslice; l++)
        {
            recon[l] -= dPOCS * tv_recon[l] / tv_norm;
        }
    }
    positivity();
}

// Return Reconstruction to Python.
Mat astra_ctvlib::getRecon(int s)
{
    return recon[s];
}

//Return the projections.
Mat astra_ctvlib::get_projections()
{
    return b;
}

// Restart the Reconstruction (Reset to Zero). 
void astra_ctvlib::restart_recon()
{
    for (int s = 0; s < Nslice; s++)
    {
        recon[s] = Mat::Zero(Ny,Nz);
    }
}

//Python functions for astra_ctvlib module.
PYBIND11_MODULE(astra_ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions using ASTRA Cuda Library";
    py::class_<astra_ctvlib> astra_ctvlib(m, "astra_ctvlib");
    astra_ctvlib.def(py::init<int,int,int,Vec>());
    astra_ctvlib.def("setTiltSeries", &astra_ctvlib::setTiltSeries, "Pass the Projections to C++ Object");
    astra_ctvlib.def("setOriginalVolume", &astra_ctvlib::setOriginalVolume, "Pass the Volume to C++ Object");
    astra_ctvlib.def("create_projections", &astra_ctvlib::create_projections, "Create Projections from Volume");
    astra_ctvlib.def("getRecon", &astra_ctvlib::getRecon, "Return the Reconstruction to Python");
    astra_ctvlib.def("initializeSART", &astra_ctvlib::initializeSART, "Generate Config File");
    astra_ctvlib.def("SART", &astra_ctvlib::SART, "ART Reconstruction");
    astra_ctvlib.def("initializeSIRT", &astra_ctvlib::initializeSIRT, "Generate Config File");
    astra_ctvlib.def("SIRT", &astra_ctvlib::SIRT, "SIRT Reconstruction");
    astra_ctvlib.def("initializeFBP", &astra_ctvlib::initializeFBP, "Generate Config File");
    astra_ctvlib.def("FBP", &astra_ctvlib::FBP, "Filtered Backprojection");
    astra_ctvlib.def("lipschits", &astra_ctvlib::lipschits, "Calculate Lipschitz Constant");
    astra_ctvlib.def("positivity", &astra_ctvlib::positivity, "Remove Negative Elements");
    astra_ctvlib.def("forwardProjection", &astra_ctvlib::forwardProjection, "Forward Projection");
    astra_ctvlib.def("copy_recon", &astra_ctvlib::copy_recon, "Copy the reconstruction");
    astra_ctvlib.def("matrix_2norm", &astra_ctvlib::matrix_2norm, "Calculate L2-Norm of Reconstruction");
    astra_ctvlib.def("vector_2norm", &astra_ctvlib::vector_2norm, "Calculate L2-Norm of Projection (aka Vectors)");
    astra_ctvlib.def("dyn_vector_2norm", &astra_ctvlib::dyn_vector_2norm, "Calculate L2-Norm of Partially Sampled Projections (aka Vectors)");
    astra_ctvlib.def("rmse", &astra_ctvlib::rmse, "Calculate reconstruction's RMSE");
    astra_ctvlib.def("tv", &astra_ctvlib::tv_3D, "Measure 3D TV");
    astra_ctvlib.def("original_tv", &astra_ctvlib::original_tv_3D, "Measure original TV");
    astra_ctvlib.def("tv_gd", &astra_ctvlib::tv_gd_3D, "3D TV Gradient Descent");
    astra_ctvlib.def("get_projections", &astra_ctvlib::get_projections, "Return the projection matrix to python");
    astra_ctvlib.def("poissonNoise", &astra_ctvlib::poissonNoise, "Add Poisson Noise to Projections");
    astra_ctvlib.def("restart_recon", &astra_ctvlib::restart_recon, "Set all the Slices Equal to Zero");
    astra_ctvlib.def("gpuCount", &astra_ctvlib::checkNumGPUs, "Check Num GPUs available");
}
