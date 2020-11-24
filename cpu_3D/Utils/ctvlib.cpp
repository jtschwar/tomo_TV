
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

    b.resize(Nslice, Nrow); g.resize(Nslice, Nrow);
    
    //Initialize all the Slices in Recon as Zero.
    recon = new Mat[Nslice]; //Final Reconstruction.
    
    // Initialize the 3D Matrices as zeros. 
    for (int i=0; i < Nslice; i++)
    {
        recon[i] = Mat::Zero(Ny, Nz);
    }
}

int ctvlib::get_Nslice(){
    return Nslice;
}

int ctvlib::get_Nray() {
    return Ny;
}

// Temporary copy for measuring 3D TV - Derivative.
void ctvlib::initialize_recon_copy() {
    temp_recon = new Mat[Nslice];
    
    #pragma omp parallel for
    for (int i = 0; i < Nslice; i++) {
        temp_recon[i] = Mat::Zero(Ny, Nz);
    }
}

// Temporary copy for measuring 3D TV - Derivative.
void ctvlib::initialize_original_volume() {
    original_volume = new Mat[Nslice];
    
    #pragma omp parallel for
    for (int i = 0; i < Nslice; i++) {
        original_volume[i] = Mat::Zero(Ny, Nz);
    }
}

// Temporary copy for measuring 3D TV - Derivative.
void ctvlib::initialize_tv_recon() {
    tv_recon = new Mat[Nslice];
    
    #pragma omp parallel for
    for (int i = 0; i < Nslice; i++) {
        tv_recon[i] = Mat::Zero(Ny, Nz);
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
        mat_slice.resize(Ny,Nz);
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
//    temp_b.array() -= b.array();
//    float std = sqrt( ( temp_b.array() - temp_b.mean() ).square().sum() / (temp_b.size() - 1) );
}

// Regular or Stochastic ART Reconstruction.
void ctvlib::ART(float beta)
{
    #pragma omp parallel for
    for (int s=0; s < Nslice; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        float a;
        for(int j = 0; j < Nrow; j++)
        {
            a = ( b(s,j) - A.row(j).dot(vec_recon) ) / innerProduct(j);
            vec_recon += A.row(j).transpose() * a * beta;
        }
        mat_slice = vec_recon;
        mat_slice.resize(Ny, Nz);
    }
    positivity();
}

// Regular or Stochastic ART Reconstruction.
void ctvlib::randART(float beta)
{
    std::vector<int> A_index = calc_proj_order(Nrow);
    
    #pragma omp parallel for
    for (int s=0; s < Nslice; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        float a;
        for(int j = 0; j < Nrow; j++)
        {
            j = A_index[j];
            a = ( b(s,j) - A.row(j).dot(vec_recon) ) / innerProduct(j);
            vec_recon += A.row(j).transpose() * a * beta;
        }
        mat_slice = vec_recon;
        mat_slice.resize(Ny, Nz);
    }
    positivity();
}

std::vector<int> ctvlib::calc_proj_order(int n)
{
    std::vector<int> a(n);
    for (int i=0; i < n; i++){ a[i] = i; }
    
    random_device rd;
    mt19937 g(rd());
    shuffle(a.begin(), a.end(), g);
    
    return a;
}

//Calculate Lipshits Gradient (for SIRT).
float ctvlib::lipschits()
{
    VectorXf f(Ncol);
    f.setOnes();
    return (A.transpose() * (A * f)).maxCoeff();
}

// SIRT Reconstruction.
void ctvlib::SIRT(float beta)
{
    #pragma omp parallel for
    for (int s=0; s < Nslice; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        vec_recon += A.transpose() * ( b.row(s).transpose() - A * vec_recon ) * beta;
        mat_slice = vec_recon;
        mat_slice.resize(Ny, Nz);
    }
    positivity();
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
    innerProduct.resize(Nrow);
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
float ctvlib::data_distance()
{
  forwardProjection();
  return (g - b).norm() / g.size(); // Nrow*Nslice,sum_{ij} M_ij^2 / Nrow*Nslice
}

// Foward project the data.
void ctvlib::forwardProjection()
{
    #pragma omp parallel for
    for (int s = 0; s < Nslice; s++)
    {
        Mat& mat_slice = recon[s];
        mat_slice.resize(mat_slice.size(),1);
        VectorXf vec_recon = mat_slice;
        for (int i=0; i < Nrow; i++)
        {
            g(s,i) = A.row(i).dot(vec_recon);
        }
        mat_slice.resize(Ny,Nz);
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
        A.coeffRef(pyA(0,i), pyA(1,i)) = pyA(2,i);
    }
    A.makeCompressed();
}

void ctvlib::update_proj_angles(Eigen::Ref<Mat> pyA) {
    Nrow = Ny * pyA.cols();
    A.resize(Nrow,Ncol);
    b.resize(Nslice, Nrow); g.resize(Nslice, Nrow);
    
    for (int i=0; i < pyA.cols(); i++) {
        A.coeffRef(pyA(0,i), pyA(1,i)) = pyA(2,i);
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
                
                tv_recon[i](j,k) = sqrt(eps + ( original_volume[i](j,k) - original_volume[ip](j,k) ) * ( original_volume[i](j,k) - original_volume[ip](j,k) )
                                        + ( original_volume[i](j,k) - original_volume[i](jp,k) ) * ( original_volume[i](j,k) - original_volume[i](jp,k) )
                                        + ( original_volume[i](j,k) - original_volume[i](j,kp) ) * ( original_volume[i](j,k) - original_volume[i](j,kp) ));

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
Mat ctvlib::getRecon(int s)
{
    return recon[s];
}

//Return the projections.
Mat ctvlib::get_projections()
{
    return b;
}

// Restart the Reconstruction (Reset to Zero). 
void ctvlib::restart_recon()
{
    for (int s = 0; s < Nslice; s++)
    {
        recon[s] = Mat::Zero(Ny,Nz);
    }
}

//Python functions for ctvlib module. 
PYBIND11_MODULE(ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions";
    py::class_<ctvlib> ctvlib(m, "ctvlib");
    ctvlib.def(py::init<int,int, int>());
    ctvlib.def("Nslice", &ctvlib::get_Nslice, "Get Nslice");
    ctvlib.def("Nray", &ctvlib::get_Nray, "Get Nray");
    ctvlib.def("set_tilt_series", &ctvlib::setTiltSeries, "Pass the Projections to C++ Object");
    ctvlib.def("initialize_recon_copy", &ctvlib::initialize_recon_copy, "Initialize Recon Copy");
    ctvlib.def("initialize_tv_recon", &ctvlib::initialize_tv_recon, "Initialize TV Recon");
    ctvlib.def("initialize_original_volume", &ctvlib::initialize_original_volume, "Initialize Original Volume");
    ctvlib.def("set_original_volume", &ctvlib::setOriginalVolume, "Pass the Volume to C++ Object");
    ctvlib.def("create_projections", &ctvlib::create_projections, "Create Projections from Volume");
    ctvlib.def("get_recon", &ctvlib::getRecon, "Return the Reconstruction to Python");
    ctvlib.def("ART", &ctvlib::ART, "ART Reconstruction");
    ctvlib.def("randART", &ctvlib::randART, "Stochastic ART Reconstruction");
    ctvlib.def("SIRT", &ctvlib::SIRT, "SIRT Reconstruction");
    ctvlib.def("lipschits", &ctvlib::lipschits, "Calculate Lipschitz Constant");
    ctvlib.def("row_inner_product", &ctvlib::normalization, "Calculate the Row Inner Product for Measurement Matrix");
    ctvlib.def("positivity", &ctvlib::positivity, "Remove Negative Elements");
    ctvlib.def("forward_projection", &ctvlib::forwardProjection, "Forward Projection");
    ctvlib.def("load_A", &ctvlib::loadA, "Load Measurement Matrix Created By Python");
    ctvlib.def("copy_recon", &ctvlib::copy_recon, "Copy the reconstruction");
    ctvlib.def("matrix_2norm", &ctvlib::matrix_2norm, "Calculate L2-Norm of Reconstruction");
    ctvlib.def("data_distance", &ctvlib::data_distance, "Calculate L2-Norm of Projection (aka Vectors)");
    ctvlib.def("rmse", &ctvlib::rmse, "Calculate reconstruction's RMSE");
    ctvlib.def("tv", &ctvlib::tv_3D, "Measure 3D TV");
    ctvlib.def("original_tv", &ctvlib::original_tv_3D, "Measure original TV");
    ctvlib.def("tv_gd", &ctvlib::tv_gd_3D, "3D TV Gradient Descent");
    ctvlib.def("get_projections", &ctvlib::get_projections, "Return the projection matrix to python");
    ctvlib.def("poisson_noise", &ctvlib::poissonNoise, "Add Poisson Noise to Projections");
    ctvlib.def("restart_recon", &ctvlib::restart_recon, "Set all the Slices Equal to Zero");
}
