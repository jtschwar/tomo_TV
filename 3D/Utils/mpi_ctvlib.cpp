//
//  tlib.cpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "mpi_ctvlib.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <cmath>
#include <random>
#include <mpi.h>

#define PI 3.14159265359

using namespace Eigen;
using namespace std;
namespace py = pybind11;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat;

mpi_ctvlib::mpi_ctvlib(int Ns, int Nray, int Nproj)
{
    //Intialize all the Member variables.
    Nslice = Ns;
    Ny = Nray;
    Nz = Nray;
    Nrow = Nray*Nproj;
    Ncol = Ny*Nz;
    A.resize(Nrow,Ncol);
    innerProduct.resize(Nrow);
    b.resize(Ny, Nrow);
    g.resize(Ny, Nrow);
    
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size = nproc; 
    
    //Calculate the number of slices for each rank.
    Nslice_loc = int(Nslice/nproc);
    if (rank < Nslice%nproc) Nslice_loc++; 
    first_slice = rank*Nslice_loc; 
    if (rank < Nslice%nproc) 
        first_slice += rank%nproc; 
    last_slice = first_slice + Nslice_loc - 1; 

    //All the rank Initialize all the 3D-matrices.
        
   //Final Reconstruction.
    recon = new Mat[Nslice_loc+2]; /* I added two more slices, Mat[Nslice_loc] on rank N = Mat[0] on rank N+1
        */
    // Temporary copy for measuring changes in TV and ART.
    temp_recon = new Mat[Nslice_loc+2];
        
    // Temporary copy for measuring 3D TV - Derivative.
    tv_recon = new Mat[Nslice_loc+2];
        
        // Original Volume for Simulation Studies.
    original_volume = new Mat[Nslice_loc+2];

        // Initialize the 3D Matrices as zeros.
    #pragma omp parallel for
    for (int i=0; i < Nslice_loc; i++)
    {
         recon[i] = Mat::Zero(Ny, Nz);
         temp_recon[i] = Mat::Zero(Ny, Nz);
         tv_recon[i] = Mat::Zero(Ny,Nz);
    }
}

//Import tilt series (projections) from Python.
void mpi_ctvlib::setTiltSeries(Mat in)
{
    b = in;
}

// Import the original volume from python.
void mpi_ctvlib::setOriginalVolume(Mat in, int slice)
{
    original_volume[slice] = in;
}

// Create projections from Volume (for simulation studies)
void mpi_ctvlib::create_projections()
{
    for (int s = 0; s < Nslice_loc; s++)
    {
        Mat& mat_slice = original_volume[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        #pragma omp parallel for
        for (int i=0; i < Nrow; i++)
        {
            b(s,i) = A.row(i).dot(vec_recon);
        }
        mat_slice.resize(Ny,Nz);
    }
}

// Add poisson noise to projections.
void mpi_ctvlib::poissonNoise(int Nc)
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
    temp_b.array() -= b.array();
    float std = sqrt( ( temp_b.array() - temp_b.mean() ).square().sum() / (temp_b.size() - 1));
}


// ART Reconstruction.
void mpi_ctvlib::ART(float beta, int dyn_ind)
{
//    
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1) { dyn_ind = Nrow; }
    //Calculate how many projections were sampled.
    else { dyn_ind *= Ny; }
    
    #pragma omp parallel for
    for (int s=0; s < Nslice_loc; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        float a;
        for(int j=0; j < dyn_ind; j++)
        {
            a = ( b(s,j) - A.row(j).dot(vec_recon) ) / innerProduct(j);
            vec_recon += A.row(j).transpose() * a * beta;
        }
        mat_slice = vec_recon;
        mat_slice.resize(Ny, Nz);
    }
}


void mpi_ctvlib::updateLeftSlice(Mat *vol) {
    /*
    Need to make sure this is OK. 
    */
    MPI_Status status;
    MPI_Send(vol[Nslice_loc-1], Ny*Nz, MPI_FLOAT, (rank+1)%nproc, MPI_ANY_TAG, MPI_COMM_WORLD); 
    MPI_Recv(vol[Nslice_loc+1], Ny*Nz, MPI_FLOAT, (rank-1+nproc)%nproc, MPI_ANY_TAB, MPI_COMM_WORLD, &status); 
}


void mpi_ctvlib::updateRightSlice(Mat *vol) {
    MPI_Status status;
    MPI_Send(vol[0], Ny*Nz, MPI_FLOAT, (rank-1+nproc)%nproc, MPI_ANY_TAG, MPI_COMM_WORLD); 
    MPI_Recv(vol[Nslice_loc], Ny*Nz, MPI_FLOAT, (rank+1)%nproc, MPI_ANY_TAB, MPI_COMM_WORLD, &status); 
}

// Stochastic ART Reconstruction.
void mpi_ctvlib::sART(float beta, int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1) { dyn_ind = Nrow; }
    //Calculate how many projections were sampled.
    else { dyn_ind *= Ny; }
    
    // Create a random permutation of indices from [0,dyn_ind].
    std::vector<int> r_ind = rand_perm(dyn_ind);
    
    #pragma omp parallel for
    for (int s=0; s < Nslice_loc; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        float a;
        for(int j=0; j < dyn_ind; j++)
        {
            j = r_ind[j];
            a = ( b(s,j) - A.row(j).dot(vec_recon) ) / innerProduct(j);
            vec_recon += A.row(j).transpose() * a * beta;
        }
        mat_slice = vec_recon;
        mat_slice.resize(Ny, Nz);
    }
}

std::vector<int> mpi_ctvlib::rand_perm(int n)
{
    std::vector<int> a(n);
    for (int i=0; i < n; i++)
    {
        a[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(a.begin(), a.end(), g);
    return a;
}

// SIRT Reconstruction.
void mpi_ctvlib::SIRT(float beta, int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1) { dyn_ind = Nrow; }
    //Calculate how many projections were sampled.
    else { dyn_ind *= Ny; }
    
    #pragma omp parallel for
    for (int s=0; s < Nslice_loc; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        vec_recon += A.transpose() * ( b.row(s).transpose() - A * vec_recon ) * beta;
        mat_slice = vec_recon;
        mat_slice.resize(Ny, Nz);
    }
    updateBoundarySlice();
}

//Calculate Lipshits Gradient (for SIRT). 
void mpi_ctvlib::lipschits()
{
    VectorXf f(Ncol);
    f.setOnes();
    float L = (A.transpose() * (A * f)).maxCoeff();
}

// Remove Negative Voxels.
void mpi_ctvlib::positivity()
{
    #pragma omp parallel for
    for(int i=0; i<Nslice_loc; i++)
    {
        recon[i] = (recon[i].array() < 0).select(0, recon[i]);
    }
}

// Row Inner Product of Measurement Matrix.
void mpi_ctvlib::normalization()
{
    #pragma omp parallel for
    for (int i = 0; i < Nrow; i++)
    {
        innerProduct(i) = A.row(i).dot(A.row(i));
    }
}

// Create Local Copy of Reconstruction. 
void mpi_ctvlib::copy_recon()
{
    memcpy(temp_recon, recon, sizeof(recon));
}

// Measure the 2 norm between temporary and current reconstruction.
float mpi_ctvlib::matrix_2norm()
{
    float L2, L2_loc;
    #pragma omp parallel for reduction(+:L2_loc)
    for (int s =0; s < Nslice_loc; s++)
    {
        L2_loc += ( recon[s].array() - temp_recon[s].array() ).square().sum();
    }
    MPI_Allreduce(&L2_loc, &L2, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(L2);
}

// Measure the 2 norm between experimental and reconstructed projections.
float mpi_ctvlib::vector_2norm()
{
    return (g - b).norm() / g.size();
}

// Measure the 2 norm for projections when data is 'dynamically' collected.
float mpi_ctvlib::dyn_vector_2norm(int dyn_ind)
{
    dyn_ind *= Ny;
    return ( g.leftCols(dyn_ind) - b.leftCols(dyn_ind) ).norm() / g.leftCols(dyn_ind).size();
}

// Foward project the data.
void mpi_ctvlib::forwardProjection(int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1) { dyn_ind = Nrow; }
    //Calculate how many projections were sampled.
    else { dyn_ind *= Ny; }
    
    #pragma omp parallel for
    for (int s = 0; s < Nslice_loc; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        for (int i=0; i < dyn_ind; i++)
        {
            g(s,i) = A.row(i).dot(vec_recon);
        }
        mat_slice.resize(Ny,Nz);
    }
    
    //TODO: Collect the data to merge into one reprojection vector (g).
}

// Measure the RMSE (simulation studies)
float mpi_ctvlib::rmse()
{
    float rmse, rmse_loc;
    #pragma omp parallel for reduction(+:rmse)
    for (int s = 0; s < Nslice_loc; s++)
    {
        rmse_loc += ( recon[s].array() - original_volume[s].array() ).square().sum();
    }

    //MPI_Reduce.
    MPI_Allreduce(&rmse_loc, &rmse, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    rmse = sqrt( rmse / (Nslice * Ny * Nz ) );
    return rmse;
}

// Load Measurement Matrix from Python.
void mpi_ctvlib::loadA(Eigen::Ref<Mat> pyA)
{
    for (int i=0; i <pyA.cols(); i++)
    {
        A.coeffRef(pyA(0,i), pyA(1,i)) = pyA(2,i);
    }
    A.makeCompressed();
    // I comment the following line out, assuming that all the ranks read pyA, therefore it does not need to do broadcast. 
    //MPI_Bcast(&A, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
}

//Measure Reconstruction's TV.
float mpi_ctvlib::tv_3D()
{
    float tv, tv_loc;
    float eps = 1e-6;
    int nx = Nslice_loc;
    int ny = Ny;
    int nz = Nz;
    updateRightSlice(recon);
    for (int i = 0; i < Nslice_loc; i++)
    {
        int ip = i+1;
        #pragma omp parallel for
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
        tv_loc+=tv_recon[i].sum();
    }
    //MPI_Reduce.
    MPI_Allreduce(&tv_loc, &tv, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return tv;
}

//Measure Original Volume's TV.
float mpi_ctvlib::original_tv_3D()
{
    float tv, tv_loc;
    float eps = 1e-6;
    int nx = Nslice_loc;
    int ny = Ny;
    int nz = Nz;
    updateRightSlice(original_volume);
    for (int i = 0; i < Nslice_loc; i++)
    {
        int ip = i+1;
        #pragma omp parallel for
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
        tv_loc+= tv_recon[i].sum();
    }
    //MPI_Reduce.
    MPI_Allreduce(&tv_loc, &tv, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return tv;
}

// TV Minimization (Gradient Descent)
void mpi_ctvlib::tv_gd_3D(int ng, float dPOCS)
{
    float eps = 1e-6;
    float tv_norm, tv_norm_loc;
    int nx = Nslice_loc;
    int ny = Ny;
    int nz = Nz;
    updateRightSlice(recon);
    updateLeftSlice(recon);
    //Calculate TV Derivative Tensor.
    for(int g=0; g < ng; g++)
    {
        for (int i = 0; i < nx; i++)
        {
            int ip = i+1
            int im = (i-1+nx+2) % (nx+2);
            #pragma omp parallel for reduction(+:tv_norm_loc)
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
                    tv_norm_loc += tv_recon[i](j,k) * tv_recon[i](j,k);
                }
            }
        }
        
        MPI_Allreduce(&tv_norm_loc, &tv_norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        tv_norm = sqrt(tv_norm);
        
        // Gradient Descent.
        #pragma omp parallel for
        for (int l = 0; l < nx; l++)
        {
            recon[l] -= dPOCS * tv_recon[l] / tv_norm;
        }
    }
    positivity();
}

// Return Reconstruction to Python.
Mat mpi_ctvlib::getRecon(int s)
{
    return recon[s];
}

//Return the projections.
Mat mpi_ctvlib::get_projections()
{
    return b;
}

void mpi_ctvlib::restart_recon()
{
    #pragma omp parallel for
    for (int s = 0; s < Nslice; s++)
    {
        recon[s] = Mat::Zero(Ny,Ny);
    }
}

//Python functions for ctvlib module. 
PYBIND11_MODULE(mpi_ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions with OpenMPI Support";
    py::class_<mpi_ctvlib> mpi_ctvlib(m, "mpi_ctvlib");
    mpi_ctvlib.def(py::init<int,int, int>());
    mpi_ctvlib.def("setTiltSeries", &mpi_ctvlib::setTiltSeries, "Pass the Projections to C++ Object");
    mpi_ctvlib.def("setOriginalVolume", &mpi_ctvlib::setOriginalVolume, "Pass the Volume to C++ Object");
    mpi_ctvlib.def("create_projections", &mpi_ctvlib::create_projections, "Create Projections from Volume");
    mpi_ctvlib.def("getRecon", &mpi_ctvlib::getRecon, "Return the Reconstruction to Python");
    mpi_ctvlib.def("ART", &mpi_ctvlib::ART, "ART Reconstruction");
    mpi_ctvlib.def("sART", &mpi_ctvlib::sART, "Stochastic ART Reconstruction");
    mpi_ctvlib.def("SIRT", &mpi_ctvlib::SIRT, "SIRT Reconstruction");
    mpi_ctvlib.def("rowInnerProduct", &mpi_ctvlib::normalization, "Calculate the Row Inner Product for Measurement Matrix");
    mpi_ctvlib.def("positivity", &mpi_ctvlib::positivity, "Remove Negative Elements");
    mpi_ctvlib.def("forwardProjection", &mpi_ctvlib::forwardProjection, "Forward Projection");
    mpi_ctvlib.def("load_A", &mpi_ctvlib::loadA, "Load Measurement Matrix Created By Python");
    mpi_ctvlib.def("copy_recon", &mpi_ctvlib::copy_recon, "Copy the reconstruction");
    mpi_ctvlib.def("matrix_2norm", &mpi_ctvlib::matrix_2norm, "Calculate L2-Norm of Reconstruction");
    mpi_ctvlib.def("vector_2norm", &mpi_ctvlib::vector_2norm, "Calculate L2-Norm of Projection (aka Vectors)");
    mpi_ctvlib.def("dyn_vector_2norm", &mpi_ctvlib::dyn_vector_2norm, "Calculate L2-Norm of Partially Sampled Projections (aka Vectors)");
    mpi_ctvlib.def("rmse", &mpi_ctvlib::rmse, "Calculate reconstruction's RMSE");
    mpi_ctvlib.def("tv", &mpi_ctvlib::tv_3D, "Measure 3D TV");
    mpi_ctvlib.def("original_tv", &mpi_ctvlib::original_tv_3D, "Measure original TV");
    mpi_ctvlib.def("tv_gd", &mpi_ctvlib::tv_gd_3D, "3D TV Gradient Descent");
    mpi_ctvlib.def("get_projections", &mpi_ctvlib::get_projections, "Return the projection matrix to python");
    mpi_ctvlib.def("poissonNoise", &mpi_ctvlib::poissonNoise, "Add Poisson Noise to Projections");
    mpi_ctvlib.def("lip", &mpi_ctvlib::lipschits, "Add Poisson Noise to Projections");
    mpi_ctvlib.def("restart_recon", &mpi_ctvlib::restart_recon, "Set all the Slices Equal to Zero");
}
