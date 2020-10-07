//
//  mpi_astra_ctlib.cpp
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "mpi_astra_ctvlib.hpp"
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
#include <mpi.h>

using namespace astra;
using namespace Eigen;
using namespace std;
namespace py = pybind11;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;

mpi_astra_ctvlib::mpi_astra_ctvlib(int Ns, int Nray, int Nproj, Vec pyAngles)
{
    //Intialize all the Member variables.
    Nslice = Ns;
    Ny = Nray;
    Nz = Nray;
    Nrow = Nray * Nproj;
    Ncol = Ny * Nz;
    b.resize(Nslice, Nrow);
    g.resize(Nslice, Nrow);
    
    // Initialize MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    //Calculate the number of slices for each rank.
    Nslice_loc = int(Nslice/nproc);
    first_slice = rank*Nslice_loc;
    if (rank < Nslice%nproc){
        Nslice_loc++;
        first_slice += rank%nproc; }
    last_slice = first_slice + Nslice_loc - 1;
    
    //Initialize all the Slices in Recon as Zero.
    recon = Matrix3D(Nslice_loc+2,Ny,Nz); //Final Reconstruction.
    
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

int mpi_astra_ctvlib::get_rank() {
    return rank;
}

int mpi_astra_ctvlib::get_nproc() {
    return nproc;
}

int mpi_astra_ctvlib::get_Nslice_loc()
{
    return Nslice_loc;
}

int mpi_astra_ctvlib::get_first_slice()
{
    return first_slice;
}

void mpi_astra_ctvlib::initilizeInitialVolume()
{
    original_volume = Matrix3D(Nslice,Ny,Nz);
}

void mpi_astra_ctvlib::initializeReconCopy()
{
    temp_recon =  Matrix3D(Nslice_loc+2,Ny,Nz); // Temporary copy for measuring changes in TV and ART.
}

void mpi_astra_ctvlib::checkNumGPUs()
{
    //int nDevices;
    cudaGetDeviceCount(&nDevices);
    cout << "NumGPUs: " << nDevices << endl;
    localDevice = rank % nDevices;
}


//Import tilt series (projections) from Python.
void mpi_astra_ctvlib::setTiltSeries(Mat in)
{
    b = in;
}

// Import the original volume from python.
void mpi_astra_ctvlib::setOriginalVolume(Mat inBuffer, int slice)
{
    original_volume.setData(inBuffer,slice);
}

void mpi_astra_ctvlib::setRecon(Mat inBuffer, int slice)
{
    recon.setData(inBuffer,slice);
}

// Create projections from Volume (for simulation studies)
void mpi_astra_ctvlib::create_projections()
{
    // Create Sinogram ASTRA Object
    sino = new CFloat32ProjectionData2D(proj_geom);

    // Create CUDA Projector
    proj = new CCudaProjector2D(proj_geom,vol_geom);
     
    // Forward Projection Operator
    algo_fp = new CCudaForwardProjectionAlgorithm();

    int sliceInd;
    for (int s=0; s < Nslice_loc; s++) {
        
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
void mpi_astra_ctvlib::poissonNoise(int Nc)
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

void mpi_astra_ctvlib::update_projection_angles(int Nproj, Vec pyAngles)
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

void mpi_astra_ctvlib::initializeSART(std::string order)
{
    projOrder = order;
    cout << "ProjectionOrder: " << projOrder << endl;
    algo_sart = new CCudaSartAlgorithm();
    algo_sart->initialize(proj,sino,vol);
    algo_sart->setConstraints(true, 0, false, 1);
    algo_sart->setGPUIndex(localDevice);
}

// ART Reconstruction.
void mpi_astra_ctvlib::SART(float beta)
{
    int Nproj = Nrow / Ny;
    algo_sart->updateProjOrder(projOrder);
    
    int sliceInd;
    for (int s=0; s < Nslice_loc; s++) {
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

void mpi_astra_ctvlib::initializeSIRT()
{
    algo_sirt = new CCudaSirtAlgorithm();
    algo_sirt->initialize(proj, sino, vol);
    algo_sirt->setConstraints(true, 0, false, 1);
    checkNumGPUs();
    cout << "Rank : " << rank << endl;
    cout << "GPUindex : " << localDevice << endl;
    algo_sirt->setGPUIndex(localDevice);
}

// SIRT Reconstruction.
void mpi_astra_ctvlib::SIRT()
{
    int sliceInd;
    for (int s=0; s < Nslice_loc; s++)
    {
        // Pass 2D Slice and Sinogram (Measurements) to ASTRA.
        sliceInd = recon.calc_index(s,0,0);
        sino->copyData((float32*) &b(s,0));
        vol->copyData( (float32*) &recon.data[sliceInd] );

        // SIRT Reconstruction
	algo_sirt->updateSlice(sino, vol);
//        algo_sirt->initialize(proj,sino,vol);
//        algo_sirt->setConstraints(true, 0, false, 1);
        algo_sirt->run(1);
        
        // Return Slice to tomo_TV
        memcpy(&recon.data[sliceInd], vol->getData(), sizeof(float)*Ny*Nz);
    }
}

void mpi_astra_ctvlib::initializeFBP(std::string filter)
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
void mpi_astra_ctvlib::FBP()
{
    E_FBPFILTER fbfFilt = convertStringToFilter(fbfFilter);
    for (int s=0; s < Nslice_loc; s++)
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
void mpi_astra_ctvlib::copy_recon()
{
    memcpy(temp_recon.data, recon.data, sizeof(float)*Nslice*Ny*Nz);
}

// Measure the 2 norm between temporary and current reconstruction.
float mpi_astra_ctvlib::matrix_2norm()
{
    float L2, L2_loc;
    
    L2_loc = cuda_euclidean_dist(recon.data, temp_recon.data, Nslice, Ny, Nz);
    
    if (nproc==1)
        return sqrt(L2_loc);
    else {
        MPI_Allreduce(&L2_loc, &L2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        return sqrt(L2);
    }
}

// Measure the 2 norm between experimental and reconstructed projections.
float mpi_astra_ctvlib::vector_2norm()
{
    float v2_loc = (g - b).norm();
    float v2;
    if (nproc==1)
        v2 = v2_loc/g.size();
    else {
        v2_loc = v2_loc*v2_loc;
        MPI_Allreduce(&v2_loc, &v2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        v2 = sqrt(v2)/Nslice/Nrow;
    }
    return v2;
}

// Measure the 2 norm for projections when data is 'dynamically' collected.
float mpi_astra_ctvlib::dyn_vector_2norm(int dyn_ind)
{
    float v2, v2_loc;
    dyn_ind *= Ny;
    v2_loc = ( g.leftCols(dyn_ind) - b.leftCols(dyn_ind) ).norm();
    if (nproc == 1)
        v2 = v2_loc / g.leftCols(dyn_ind).size();
    else
        v2_loc = v2_loc * v2_loc;
        MPI_Allreduce(&v2_loc, &v2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        v2 = sqrt(v2)/dyn_ind/Nrow;
    return v2;
}

// Foward project the data.
void mpi_astra_ctvlib::forwardProjection()
{
    for (int s=0; s < Nslice_loc; s++) {
        
        vol->copyData((float32*) &recon.data[recon.calc_index(s,0,0)]);

        // Forward Project
        algo_fp->initialize(proj,vol,sino);
        algo_fp->run();

        memcpy(&g(s,0), sino->getData(), sizeof(float)*Nrow);
    }
}

// Measure the RMSE (simulation studies)
float mpi_astra_ctvlib::rmse()
{
    float rmse, rmse_loc;
    rmse_loc = 0.0;
    
    rmse_loc = cuda_rmse(recon.data, original_volume.data, Nslice, Ny, Nz);
    
    // MPI Reduce.
    if (nproc == 1) { rmse = rmse_loc; }
    else {
        MPI_Allreduce(&rmse_loc, &rmse, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }
    rmse = sqrt( rmse / (Nslice * Ny * Nz) );
    return rmse;
}

void mpi_astra_ctvlib::updateLeftSlice(Matrix3D vol) {
    MPI_Request request;
    int tag = 0;
    if (nproc>1) {
        MPI_Isend(&vol.data[vol.calc_index(Nslice_loc-1,0, 0)], Ny*Nz, MPI_FLOAT, (rank+1)%nproc, tag, MPI_COMM_WORLD, &request);
        MPI_Recv(&vol.data[vol.calc_index(Nslice_loc+1,0, 0)], Ny*Nz, MPI_FLOAT, (rank-1+nproc)%nproc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        MPI_Wait(&request, MPI_STATUS_IGNORE); }
    else {
        // vol[Nslice_loc+1] = vol[Nslice_loc-1]
        memcpy(&vol.data[vol.calc_index(Nslice_loc+1,0,0)], &vol.data[vol.calc_index(Nslice_loc-1,0,0)], sizeof(float)*Ny*Nz);
    }
}


void mpi_astra_ctvlib::updateRightSlice(Matrix3D vol) {
    MPI_Request request;
    int tag = 0;
    if (nproc>1) {
        MPI_Isend(&vol.data[vol.calc_index(0,0, 0)], Ny*Nz, MPI_FLOAT, (rank+1)%nproc, tag, MPI_COMM_WORLD, &request);
        MPI_Recv(&vol.data[vol.calc_index(Nslice_loc,0, 0)], Ny*Nz, MPI_FLOAT, (rank-1+nproc)%nproc, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    else {
        // vol[Nslice_loc] = vol[0]
        memcpy(&vol.data[vol.calc_index(Nslice_loc,0,0)], &vol.data[vol.calc_index(0,0,0)], sizeof(float)*Ny*Nz);
    }
}

//Measure Original Volume's TV.
float mpi_astra_ctvlib::original_tv_3D()
{
    float rmse, rmse_loc;
    updateRightSlice(original_volume);

    float tv_norm, tv_norm_loc;
    tv_norm_loc = cuda_tv_3D(original_volume.data, Nslice_loc, Ny, Nz, localDevice);

    if (nproc == 1) { return sqrt(tv_norm_loc); }
    else { 
        MPI_Allreduce(&tv_norm_loc, &tv_norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        return sqrt(tv_norm_loc);
    }
}

// TV Minimization (Gradient Descent)
float mpi_astra_ctvlib::tv_gd_3D(int ng, float dPOCS)
{
    cout << "Updating Right Slice" << endl;
    updateRightSlice(recon);
    cout << "Updating Left Slice" << endl;
    updateLeftSlice(recon);

    float tv_norm, tv_norm_loc;
    tv_norm_loc = cuda_tv_gd_3D(recon.data, ng, dPOCS, Nslice_loc, Ny, Nz, localDevice);

    if (nproc == 1) { return sqrt(tv_norm_loc); }
    else {
        MPI_Allreduce(&tv_norm_loc, &tv_norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        return sqrt(tv_norm);
    }

}

// Return Reconstruction to Python.
Mat mpi_astra_ctvlib::getRecon(int slice)
{
    return recon.getData(slice);
}

//Return the projections.
Mat mpi_astra_ctvlib::get_projections()
{
    return b;
}
                           

Mat mpi_astra_ctvlib::get_model_projections()
{
    return g;
}

// Restart the Reconstruction (Reset to Zero). 
void mpi_astra_ctvlib::restart_recon()
{
    memset(recon.data, 0, sizeof(float)*Nslice*Ny*Nz);
}

//Python functions for mpi_astra_ctvlib module.
PYBIND11_MODULE(mpi_astra_ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions using ASTRA Cuda Library";
    py::class_<mpi_astra_ctvlib> mpi_astra_ctvlib(m, "mpi_astra_ctvlib");
    mpi_astra_ctvlib.def(py::init<int,int,int,Vec>());
    mpi_astra_ctvlib.def("initilizeInitialVolume", &mpi_astra_ctvlib::initilizeInitialVolume, "Initialize Original Data");
    mpi_astra_ctvlib.def("getNsliceLoc", &mpi_astra_ctvlib::get_Nslice_loc, "Get Nslice_loc");
    mpi_astra_ctvlib.def("getFirstSlice", &mpi_astra_ctvlib::get_first_slice, "Get first_slice");
    mpi_astra_ctvlib.def("setTiltSeries", &mpi_astra_ctvlib::setTiltSeries, "Pass the Projections to C++ Object");
    mpi_astra_ctvlib.def("setOriginalVolume", &mpi_astra_ctvlib::setOriginalVolume, "Pass the Volume to C++ Object");
    mpi_astra_ctvlib.def("create_projections", &mpi_astra_ctvlib::create_projections, "Create Projections from Volume");
    mpi_astra_ctvlib.def("getRecon", &mpi_astra_ctvlib::getRecon, "Return the Reconstruction to Python");
    mpi_astra_ctvlib.def("setRecon", &mpi_astra_ctvlib::setRecon, "Return the Reconstruction to Python");
    mpi_astra_ctvlib.def("initializeSART", &mpi_astra_ctvlib::initializeSART, "Generate Config File");
    mpi_astra_ctvlib.def("SART", &mpi_astra_ctvlib::SART, "ART Reconstruction");
    mpi_astra_ctvlib.def("initializeSIRT", &mpi_astra_ctvlib::initializeSIRT, "Generate Config File");
    mpi_astra_ctvlib.def("SIRT", &mpi_astra_ctvlib::SIRT, "SIRT Reconstruction");
    mpi_astra_ctvlib.def("initializeFBP", &mpi_astra_ctvlib::initializeFBP, "Generate Config File");
    mpi_astra_ctvlib.def("FBP", &mpi_astra_ctvlib::FBP, "Filtered Backprojection");
    mpi_astra_ctvlib.def("forwardProjection", &mpi_astra_ctvlib::forwardProjection, "Forward Projection");
    mpi_astra_ctvlib.def("copy_recon", &mpi_astra_ctvlib::copy_recon, "Copy the reconstruction");
    mpi_astra_ctvlib.def("matrix_2norm", &mpi_astra_ctvlib::matrix_2norm, "Calculate L2-Norm of Reconstruction");
    mpi_astra_ctvlib.def("vector_2norm", &mpi_astra_ctvlib::vector_2norm, "Calculate L2-Norm of Projection (aka Vectors)");
    mpi_astra_ctvlib.def("dyn_vector_2norm", &mpi_astra_ctvlib::dyn_vector_2norm, "Calculate L2-Norm of Partially Sampled Projections (aka Vectors)");
    mpi_astra_ctvlib.def("rmse", &mpi_astra_ctvlib::rmse, "Calculate reconstruction's RMSE");
    mpi_astra_ctvlib.def("original_tv", &mpi_astra_ctvlib::original_tv_3D, "Measure original TV");
    mpi_astra_ctvlib.def("tv_gd", &mpi_astra_ctvlib::tv_gd_3D, "3D TV Gradient Descent");
    mpi_astra_ctvlib.def("get_projections", &mpi_astra_ctvlib::get_projections, "Return the projection matrix to python");
    mpi_astra_ctvlib.def("get_model_projections", &mpi_astra_ctvlib::get_model_projections, "Return Reprojections to Python");
    mpi_astra_ctvlib.def("poissonNoise", &mpi_astra_ctvlib::poissonNoise, "Add Poisson Noise to Projections");
    mpi_astra_ctvlib.def("restart_recon", &mpi_astra_ctvlib::restart_recon, "Set all the Slices Equal to Zero");
    mpi_astra_ctvlib.def("gpuCount", &mpi_astra_ctvlib::checkNumGPUs, "Check Num GPUs available");
}
