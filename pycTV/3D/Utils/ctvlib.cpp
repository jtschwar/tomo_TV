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
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <iostream>

#define PI 3.14159265359

using namespace Eigen;
using namespace std;
namespace py = pybind11;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat;

ctvlib::ctvlib(int Nslice, int Nray, int Nproj)
{
    // 
    Nx = Nray;
    Nrow = Nray*Nproj;
    Ncol = Nray*Nray;
    A.resize(Nrow,Ncol);
    innerProduct.resize(Nrow);
    b.resize(Nrow, Nx);
}

void ctvlib::setTiltSeries(Mat in)
{
    b = in;
}

Mat ctvlib::ART(Eigen::Ref<Eigen::VectorXf> recon, double beta, int s, int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1)
        dyn_ind = Nrow;
    else //Calculate how many projections were sampled.
        dyn_ind *= Nx;
    
    float a;
    for(int j=0; j < dyn_ind; j++)
    {
        a = ( b(s,j) - A.row(j).dot(recon) ) / innerProduct(j);
        recon += A.row(j).transpose() * a * beta;
    }
    Mat mat_recon = recon;
    mat_recon.resize(Nx, Nx);
    return mat_recon;
}

Mat ctvlib::ART2(Eigen::Ref<Eigen::VectorXf> recon, Eigen::Ref<Eigen::VectorXf> beta, int s, int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1)
        dyn_ind = Nrow;
    else //Calculate how many projections were sampled.
        dyn_ind *= Nx;
    
    float a;
    for(int j=0; j < dyn_ind; j++)
    {
        a = ( b(s,j) - A.row(j).dot(recon) ) / innerProduct(j);
        recon += A.row(j).transpose() * a * beta(j);
    }
    Mat mat_recon = recon;
    mat_recon.resize(Nx, Nx);
    return mat_recon;
}

Mat ctvlib::SIRT(Eigen::Ref<Eigen::VectorXf> recon, double beta, int s)
{
    recon += A.transpose() * ( b.row(s).transpose() - A * recon ) * beta;
    Mat mat_recon = recon;
    mat_recon.resize(Nx, Nx);
    return mat_recon;
}

void ctvlib::normalization()
{
    // Row Inner Product: RIP
    for (int i = 0; i < Nrow; i++)
    {
        innerProduct(i) = A.row(i).dot(A.row(i));
    }
}

Eigen::VectorXf ctvlib::forwardProjection(Eigen::Ref<Eigen::VectorXf> recon, int dyn_ind)
{
    //No dynamic reconstruction, assume fully sampled batch.
    if (dyn_ind == -1)
        dyn_ind = Nrow;
    else //Calculate how many projections were sampled.
        dyn_ind *= Nx;
    
    VectorXf g(dyn_ind);
    for (int i = 0; i < dyn_ind; i++)
    {
        g(i) = A.row(i).dot(recon);
    }
    return g;
}

void ctvlib::loadA(Eigen::Ref<Mat> pyA)
{
    for (int i=0; i <pyA.cols(); i++)
    {
        A.insert(pyA(0,i), pyA(1,i)) = pyA(2,i);
    }
    A.makeCompressed();
}


PYBIND11_MODULE(ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions";
    py::class_<ctvlib> ctvlib(m, "ctvlib");
    ctvlib.def(py::init<int,int, int>());
    ctvlib.def("setTiltSeries", &ctvlib::setTiltSeries, "Pass the Projections to C++ Object");
    ctvlib.def("ART", &ctvlib::ART, "ART Reconstruction");
    ctvlib.def("ART2", &ctvlib::ART2, "Dynamic ART Reconstruction");
    ctvlib.def("SIRT", &ctvlib::SIRT, "SIRT Reconstruction");
    ctvlib.def("rowInnerProduct", &ctvlib::normalization, "Calculate the Row Inner Product for Measurement Matrix");
    ctvlib.def("forwardProjection", &ctvlib::forwardProjection, "Forward Project the Reconstructions");
    ctvlib.def("create_measurement_matrix", &ctvlib::loadA, "Load Measurement Matrix Created By Python");
}


