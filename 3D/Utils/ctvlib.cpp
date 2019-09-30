//
//  tlib.cpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "ctvlib.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <cmath>
#include <random>

#define PI 3.14159265359

using namespace Eigen;
using namespace std;
namespace py = pybind11;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat;

ctvlib::ctvlib(int Ns, int Nray, int Nproj)
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
    
    //Initialize all the Slices in Recon as Zero.
    recon = new Mat[Nslice]; //Final Reconstruction.
    temp_recon = new Mat[Nslice]; // Temporary copy for measuring changes in TV and ART.
    tv_recon = new Mat[Nslice]; // Temporary copy for measuring 3D TV - Derivative. 
    original_volume = new Mat[Nslice]; // Original Volume for Simulation Studies.
    
    // Initialize the 3D Matrices as zeros. 
    for (int i=0; i < Nslice; i++)
    {
        recon[i] = Mat::Zero(Ny, Ny);
        temp_recon[i] = Mat::Zero(Ny, Ny);
        tv_recon[i] = Mat::Zero(Ny,Ny);
    }
}

//Import tilt series (projections) from Python.
void ctvlib::setTiltSeries(Mat in)
{
    b = in;
}

// Import the original volume from python.
void ctvlib::setOriginalVolume(Mat in, int slice)
{
    original_volume[slice] = in;
}

// Create projections from Volume (for simulation studies)
void ctvlib::create_projections()
{
    #pragma omp parallel for
    for (int s = 0; s < Nslice; s++)
    {
        Mat& mat_slice = original_volume[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        for (int i=0; i < Nrow; i++)
        {
            b(s,i) = A.row(i).dot(vec_recon);
        }
        mat_slice.resize(Ny,Ny);
    }
}

// Add poisson noise to projections.
void ctvlib::poissonNoise(int Nc)
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
    float std = sqrt( ( temp_b.array() - temp_b.mean() ).square().sum() / (temp_b.size() - 1) );
}

// ART Reconstruction.
void ctvlib::ART(double beta, int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1) { dyn_ind = Nrow; }
    //Calculate how many projections were sampled.
    else { dyn_ind *= Ny; }
    
    #pragma omp parallel for
    for (int s=0; s < Nslice; s++)
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
        mat_slice.resize(Ny, Ny);
    }
}

// SIRT Reconstruction.
void ctvlib::SIRT(double beta, int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1) { dyn_ind = Nrow; }
    //Calculate how many projections were sampled.
    else { dyn_ind *= Ny; }
    
    #pragma omp parallel for
    for (int s=0; s < Nslice; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        vec_recon += A.transpose() * ( b.row(s).transpose() - A * vec_recon ) * beta;
        mat_slice = vec_recon;
        mat_slice.resize(Ny, Ny);
    }
}

// Remove Negative Voxels.
void ctvlib::positivity()
{
    #pragma omp parallel for
    for(int i=0; i<Nslice; i++)
    {
        recon[i] = (recon[i].array() < 0).select(0, recon[i]);
    }
}

// Row Inner Product of Measurement Matrix.
void ctvlib::normalization()
{
    #pragma omp parallel for
    for (int i = 0; i < Nrow; i++)
    {
        innerProduct(i) = A.row(i).dot(A.row(i));
    }
}

// Create Local Copy of Reconstruction. 
void ctvlib::copy_recon()
{
    memcpy(temp_recon, recon, sizeof(recon));
}

// Measure the 2 norm between temporary and current reconstruction.
float ctvlib::matrix_2norm()
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
float ctvlib::vector_2norm()
{
    return (g - b).norm() / g.size();
}

// Measure the 2 norm for projections when data is 'dynamically' collected.
float ctvlib::dyn_vector_2norm(int dyn_ind)
{
    dyn_ind *= Ny;
    return ( g.leftCols(dyn_ind) - b.leftCols(dyn_ind) ).norm() / g.leftCols(dyn_ind).size();
}

// Foward project the data.
void ctvlib::forwardProjection(int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1) { dyn_ind = Nrow; }
    //Calculate how many projections were sampled.
    else { dyn_ind *= Ny; }
    
    #pragma omp parallel for
    for (int s = 0; s < Nslice; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        for (int i=0; i < dyn_ind; i++)
        {
            g(s,i) = A.row(i).dot(vec_recon);
        }
        mat_slice.resize(Ny,Ny);
    }
}

// Measure the RMSE (simulation studies)
float ctvlib::rmse()
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

// Load Measurement Matrix from Python.
void ctvlib::loadA(Eigen::Ref<Mat> pyA)
{
    for (int i=0; i <pyA.cols(); i++)
    {
        A.insert(pyA(0,i), pyA(1,i)) = pyA(2,i);
    }
    A.makeCompressed();
}

//Measure Reconstruction's TV.
float ctvlib::tv_3D()
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
                tv_recon[i](j,k) = sqrt(eps + pow( recon[i](j,k) - recon[ip](j,k) , 2)
                                        + pow( recon[i](j,k) - recon[i](jp,k) , 2)
                                        + pow( recon[i](j,k) - recon[ip](j,kp) , 2));
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
float ctvlib::original_tv_3D()
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
                                        + pow( original_volume[i](j,k) - original_volume[ip](j,kp) , 2));
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
void ctvlib::tv_gd_3D(int ng, float dPOCS)
{
    float eps = 1e-8;
    int nx = Nslice;
    int ny = Ny;
    int nz = Nz;
    
    //Calculate TV Derivative Tensor.
    for(int g=0; g < ng; g++)
    {
        #pragma omp parallel for
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
                    float v1d = sqrt(eps + pow( recon[i](j, k) - recon[ip](j, k) , 2)
                                      +  pow( recon[i](j, k) - recon[i](jp, k) , 2)
                                      +  pow( recon[i](j, k) - recon[i](j, kp) , 2));
                    float v2n = recon[i](j, k) - recon[im](j, k);
                    float v2d = sqrt(eps + pow( recon[im](j, k) - recon[i](j, k) , 2)
                                      +  pow( recon[im](j, k) - recon[im](jp, k) , 2)
                                      +  pow( recon[im](j, k) - recon[im](j, kp) , 2));
                    float v3n = recon[i](j, k) - recon[i](jm, k);
                    float v3d = sqrt(eps + pow( recon[i](jm, k) - recon[ip](jm, k) , 2)
                                      +  pow( recon[i](jm, k) - recon[i](j, k) , 2)
                                      +  pow( recon[i](jm, k) - recon[i](jm, kp) , 2));
                    float v4n = recon[i](j, k) - recon[i](j, km);
                    float v4d = sqrt(eps + pow( recon[i](j, km) - recon[ip](j, km) , 2)
                                      + pow( recon[i](j, km) - recon[i](jp, km) , 2)
                                      + pow( recon[i](j, km) - recon[i](j, k) , 2));
                    tv_recon[i](j,k) = v1n/v1d + v2n/v2d + v3n/v3d + v4n/v4d;
                }
            }
        }
        
        // Normalize TV Gradient Tensor.
        float tv_norm;
        #pragma omp parallel for reduction(+:tv_norm)
        for (int s =0; s < Nslice; s++)
        {
            tv_norm += tv_recon[s].array().square().sum();
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
Mat ctvlib::getRecon(int s)
{
    return recon[s];
}

//Return the projections.
Mat ctvlib::get_projections()
{
    return b;
}

// Return a 2D slice to python.
Mat ctvlib::get_slice(int s)
{
    return recon[s];
}

//Python functions for ctvlib module. 
PYBIND11_MODULE(ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions";
    py::class_<ctvlib> ctvlib(m, "ctvlib");
    ctvlib.def(py::init<int,int, int>());
    ctvlib.def("setTiltSeries", &ctvlib::setTiltSeries, "Pass the Projections to C++ Object");
    ctvlib.def("setOriginalVolume", &ctvlib::setOriginalVolume, "Pass the Volume to C++ Object");
    ctvlib.def("create_projections", &ctvlib::create_projections, "Create Projections from Volume");
    ctvlib.def("getRecon", &ctvlib::getRecon, "Return the Reconstruction to Python");
    ctvlib.def("ART", &ctvlib::ART, "ART Reconstruction");
    ctvlib.def("SIRT", &ctvlib::SIRT, "SIRT Reconstruction");
    ctvlib.def("rowInnerProduct", &ctvlib::normalization, "Calculate the Row Inner Product for Measurement Matrix");
    ctvlib.def("positivity", &ctvlib::positivity, "Remove Negative Elements");
    ctvlib.def("forwardProjection", &ctvlib::forwardProjection, "Forward Projection");
    ctvlib.def("load_A", &ctvlib::loadA, "Load Measurement Matrix Created By Python");
    ctvlib.def("copy_recon", &ctvlib::copy_recon, "Copy the reconstruction");
    ctvlib.def("matrix_2norm", &ctvlib::matrix_2norm, "Calculate L2-Norm of Reconstruction");
    ctvlib.def("vector_2norm", &ctvlib::vector_2norm, "Calculate L2-Norm of Projection (aka Vectors)");
    ctvlib.def("dyn_vector_2norm", &ctvlib::dyn_vector_2norm, "Calculate L2-Norm of Partially Sampled Projections (aka Vectors)");
    ctvlib.def("rmse", &ctvlib::rmse, "Calculate reconstruction's RMSE");
    ctvlib.def("tv", &ctvlib::tv_3D, "Measure 3D TV");
    ctvlib.def("original_tv", &ctvlib::original_tv_3D, "Measure original TV");
    ctvlib.def("tv_gd", &ctvlib::tv_gd_3D, "3D TV Gradient Descent");
    ctvlib.def("get_projections", &ctvlib::get_projections, "Return the projection matrix to python");
    ctvlib.def("poissonNoise", &ctvlib::poissonNoise, "Add Poisson Noise to Projections");
    ctvlib.def("get_slice", &ctvlib::get_slice, "Get Slice of Recon");
}
