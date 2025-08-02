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

#define PI 3.14159265359

using namespace Eigen;
using namespace std;
namespace py = pybind11;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor> SpMat;

ctvlib::ctvlib(int Nx, int Ny)
{
    Nrow = Nx;
    Ncol = Ny;
    A.resize(Nrow,Ncol);
    innerProduct.resize(Nrow);
}

void ctvlib::loadA(Eigen::Ref<Mat> pyA)
{
    for (int i=0; i <pyA.cols(); i++)
    {
        A.insert(pyA(0,i), pyA(1,i)) = pyA(2,i);
    }
    A.makeCompressed();
}

SpMat ctvlib::parallelRay(int Nray, Eigen::VectorXf angles)
{

    //Nside = Nray = y dimension of tilt series.
    int Nside = Nray;

    float pixelWidth = 1;
    float rayWidth = 1;

    //Number of projections.
    int Nproj = angles.rows();
    int idxend = 0;

    //Initialize vectors that contain matrix elements and corresponding row/column numbers.
    //Ray coordinates at 0 degrees.
    VectorXf offsets = VectorXf::LinSpaced( Nray + 1, -(Nray - 1) * 0.5, (Nray + 1) * 0.5 ) * rayWidth;
    VectorXf xgrid = VectorXf::LinSpaced( Nray + 1, -Nray * 0.5 , Nray * 0.5) * pixelWidth;
    VectorXf ygrid = VectorXf::LinSpaced( Nray + 1, - Nray * 0.5 , Nray * 0.5 ) * pixelWidth;

    //Initialize vectors that contain matrix elements and corresponding row/column numbers
    long int max_elements = 2 * Nside * Nproj * Nray;
    VectorXf rows(max_elements), cols(max_elements), vals(max_elements);

    //Loop over projection angles.
    for(int i=0; i < Nproj; i++)
    {
        float ang = angles(i) * PI / 180;
        //Points passed by rays at current angles.
        VectorXf xrayRoated = cos(ang) * offsets;
        VectorXf yrayRoated = sin(ang) * offsets;
        xrayRoated = (xrayRoated.array().abs() < 1e-8).select(0, xrayRoated);
        yrayRoated = (yrayRoated.array().abs() < 1e-8).select(0, yrayRoated);

        float a = -sin(ang);
        a = rmepsilon(a);
        float b = cos(ang);
        b = rmepsilon(b);

        //Loop rays in current projection.
        for(int j = 0; j < Nray; j++ )
        {
            //Ray: y = tx * x + intercept.
            VectorXf t_xgrid, y_xgrid, t_ygrid, x_ygrid;
            t_xgrid = (xgrid.array() - xrayRoated(j)) / a;
            y_xgrid = b * t_xgrid.array() + yrayRoated(j);
            t_ygrid = (ygrid.array() - yrayRoated(j)) / b;
            x_ygrid = a * t_ygrid.array() + xrayRoated(j);

            //Collect all points
            long tne = t_xgrid.size() + t_ygrid.size(); // tne = total number of elements
            VectorXf t_grid(tne), xx_temp(tne), yy_temp(tne);
            t_grid << t_xgrid, t_ygrid;
            xx_temp << xgrid, x_ygrid;
            yy_temp << y_xgrid, ygrid;

            //Sort the coordinates according to intersection time.
            VectorXf I = VectorXf::LinSpaced(t_grid.size(), 0, t_grid.size()-1);
            std::sort(I.data(), I.data() + I.size(), [&t_grid] (size_t i1, size_t i2) {return t_grid[i1] < t_grid[i2];} );
            VectorXf xx(I.size()), yy(I.size());
            for(int k=0; k<t_grid.size(); k++){
                xx(k) = xx_temp(I(k));
                yy(k) = yy_temp(I(k));
            }

            // Get rid of points that are outside the image grid.
            I.resize(xx.size()), I.setZero();
            float vol_boundary = Nside/2.0 * pixelWidth;
            for(int k=0; k<xx.size(); k++)
            {
                if(xx(k) >= -vol_boundary && xx(k) <= vol_boundary)
                {    if(yy(k) >= -vol_boundary && yy(k) <= vol_boundary)
                     {
                        I(k) = 1.0;
                     }

                }
            }
            removeBadElements(xx, yy, I);

            //If the ray pass through the image grid
            if (xx.size() != 0 && yy.size() != 0)
                //Get rid of double counted points.
                I.resize(xx.size()), I.setOnes();
                for(int k=0; k<xx.size()-1; k++)
                {
                    if(abs(xx(k+1) - xx(k)) <= 1e-4)
                    {    if(abs(yy(k+1) - yy(k)) <= 1e-4)
                         {
                             I(k) = 0;
                         }
                    }
                }
                removeBadElements(xx, yy, I);

                //Calculate the length within the cell.
                tne = xx.size() - 1;
                VectorXf length(tne);
                for(int k=0; k<tne; k++)
                {
                    length(k) = sqrt( pow((xx(k+1) - xx(k)),2) + pow((yy(k+1) - yy(k)),2) );
                }
                int numvals = length.size();

                //Remove the rays that are on the boundary of the box in the
                //top or to the right of the image grid
                bool check1, check2, check;
                check1 = check2 = false;
                check = true;
                if (b == 0 && abs(yrayRoated(j) - Nside/2 * pixelWidth) < 1e-15) {
                    check1 = true;
                }
                if (a == 0 && abs(xrayRoated(j) - Nside/2 * pixelWidth) < 1e-15) {
                    check2 = true;
                }
                if (check1 || check2) {
                    check = false;
                }

                //Calculate corresponding indices in measurement matrix
                if(numvals > 0 && check == true)
                {

                    tne = xx.size() - 1;
                    VectorXf midpointsX(tne);
                    VectorXf midpointsY(tne);
                    for(int k=0; k<tne; k++)
                    {
                        midpointsX(k) = 0.5 * (xx(k) + xx(k+1));
                        midpointsY(k) = 0.5 * (yy(k) + yy(k+1));
                    }

                    midpointsX = (midpointsX.array().abs() < 1e-10).select(0, midpointsX);
                    midpointsY = (midpointsY.array().abs() < 1e-10).select(0, midpointsY);

                    //Calculate the pixel index for mid points
                    VectorXf pixelIndex(tne);
                    pixelIndex = floor((Nside/2 - midpointsY.array()/pixelWidth))*Nside + floor((midpointsX.array()/pixelWidth + Nside/2));
                    pixelIndex = (pixelIndex.array() < 0 ).select(0,pixelIndex);

                    //Create the indices to store the values to the measurement matrix
                    int idxstart = idxend;
                    idxend = idxstart + numvals;
                    int idx = 0;

                    //Store row numbers, column numbers and values
                    for(int k=idxstart; k<idxend; k++)
                    {
                        rows(k) = i * Nray + j;
                        cols(k) = pixelIndex(idx);
                        vals(k) = length(idx);
                        idx = idx + 1;
                    }
                }
        }

    }

    //Truncate excess zeros.
    for(int i=0; i <idxend; i++)
    {
        A.insert(rows(i), cols(i)) = vals(i);
    }
    A.makeCompressed();

    return A;
}

void ctvlib::ART(Eigen::Ref<Eigen::VectorXf> recon, Eigen::Ref<Eigen::VectorXf> b, double beta)
{

    float a;
    int Nrow = A.rows();
    for(int j=0; j < Nrow; j++)
    {
        a = (b(j) - A.row(j).dot(recon)) / innerProduct(j);
        recon += A.row(j).transpose() * a * beta;
    }
}

void ctvlib::ART2(Eigen::Ref<Eigen::VectorXf> recon, Eigen::Ref<Eigen::VectorXf> b, double beta, int max_row)
{
    
    float a;
    int Nrow = A.rows();
    for(int j=0; j < max_row; j++)
    {
        a = (b(j) - A.row(j).dot(recon)) / innerProduct(j);
        recon += A.row(j).transpose() * a * beta;
    }
}

void ctvlib::SIRT(Eigen::Ref<Eigen::VectorXf> recon, Eigen::Ref<Eigen::VectorXf> b, double beta)
{
    recon += A.transpose() * ( b - A * recon ) * beta;
}


float ctvlib::rmepsilon(float input)
{
    if (abs(input) < 1e-10)
        input = 0;
    return input;
}

void ctvlib::removeBadElements(Eigen::VectorXf& xx, Eigen::VectorXf& yy, Eigen::VectorXf I)
{
    //Remove elements at indices where I == 0.
    VectorXf xx_temp(xx.size()), yy_temp(xx.size());
    int ind = 0;
    for(int k=0; k<xx.size();k++)
    {
        if(I(k) != 0)
        {
            xx_temp(ind) = xx(k);
            yy_temp(ind) = yy(k);
            ind++;
        }
    }
    xx.resize(ind), yy.resize(ind);
    xx = xx_temp.head(ind);
    yy = yy_temp.head(ind);
}

void ctvlib::normalization()
{
    // Row Inner Product: RIP
    for (int i = 0; i < Nrow; i++)
    {
        innerProduct(i) = A.row(i).dot(A.row(i));
    }
}

Eigen::MatrixXf ctvlib::tv_loop(Eigen::MatrixXf& recon, float dPOCS, int ng)
{
    MatrixXf v;
    for (int i = 0; i < ng; i++)
    {
        v = tv2Dderivative(recon);
        v /= v.norm();
        recon.array() -= dPOCS * v.array();
    }
    return recon;
}

Eigen::MatrixXf ctvlib::tv2Dderivative(Eigen::MatrixXf recon)
{
    int padX = recon.rows() + 2;
    int padY = recon.cols() + 2;
    
    MatrixXf v1n, v1d, v2n, v2d, v3n, v3d, v;
    MatrixXf r(padX, padY), rXp(padX, padY), rYp(padX, padY);
    r.setZero(), rXp.setZero(), rYp.setZero();
    circshift(recon, r, 0, 0), circshift(recon, rXp, 1, 0), circshift(recon, rYp, 0, 1);
    v1n = 4 * r.array() - 2 * rXp.array() - 2 * rYp.array();
    v1d = sqrt( 1e-8 + (r - rXp).array().square() + (r - rYp).array().square() );
    rXp.resize(0,0), rYp.resize(0,0);
    
    MatrixXf rXn(padX, padY), rXnYp(padX, padY);
    rXn.setZero(), rXnYp.setZero();
    circshift(recon, rXn, -1, 0), circshift(recon, rXnYp, -1, 1);
    v2n = 2 * (r - rXn).array();
    v2d = sqrt( 1e-8 + (rXn - r).array().square() + (rXn - rXnYp).array().square() );
    rXn.resize(0,0), rYp.resize(0,0);
    
    MatrixXf rYn(padX, padY), rXpYn(padX, padY);
    rYn.setZero(), rXpYn.setZero();
    circshift(recon, rYn, 0, -1), circshift(recon, rXpYn, 1, -1);
    v3n = 2 * (r - rYn).array();
    v3d = sqrt( 1e-8 + (rYn - r).array().square() + (rYn - rXpYn).array().square() );
    rYn.resize(0,0), rXpYn.resize(0,0);
    
    v = v1n.array() / v1d.array() + v2n.array() / v2d.array() + v3n.array() / v3d.array();
    MatrixXf v2(recon.rows(), recon.cols());
    v2 = v.block(1, 1, recon.rows(), recon.cols());
    v.resize(0,0), v1n.resize(0,0), v1d.resize(0,0), v2n.resize(0,0), v2d.resize(0,0), v3n.resize(0,0), v3d.resize(0,0);
    return v2;
}

void ctvlib::circshift(Eigen::MatrixXf input, Eigen::MatrixXf& output, int i, int j)
{
    // i == shift in the x - direction.
    // j == shift in the y - direction.
    output.block(1+i, 1+j, input.rows(), input.rows()) = input;
}

Eigen::VectorXf ctvlib::forwardProjection(Eigen::Ref<Eigen::VectorXf> recon, int max_row)
{
    Eigen::VectorXf g;

    if (max_row == Nrow)
    {
        g = A * recon;
    }
    else
    {
        g = A.topRows(max_row) * recon;
    }
    return g;
}

float ctvlib::CosAlpha(Eigen::MatrixXf& recon,  Eigen::VectorXf& b, Eigen::VectorXf& g, int max_row)
{
    float cosA, norm;
    int Nx, Ny;
    Eigen::MatrixXf tv_derivative;
    tv_derivative = tv2Dderivative(recon);
    
    Nx = recon.rows();
    Ny = recon.cols();
    
    MatrixXf d_tv(Ncol,1), d_data(Ncol,1);
    d_tv.setZero(), d_data.setZero();

    //Vectorize.
    tv_derivative.resize(Ncol,1);

    VectorXf nabla_h(Ncol);
    nabla_h = 2 * A.topRows(max_row).transpose() * ( g - b.topRows(max_row) );
    for(int i=0; i < recon.size(); i++ )
    {
        if( abs(recon(i)) > 1e-10 )
        {
            d_tv(i) = tv_derivative(i);
            d_data(i) = nabla_h(i);
            cosA += d_data(i) * d_tv(i);
        }
    }
    recon.resize(Nx, Ny);
    norm = d_data.norm() * d_tv.norm();
    cosA /= norm;
    return cosA;
}

void ctvlib::poissonNoise(Eigen::VectorXf& b, int Nc)
{
    VectorXf temp_b(b.size());
    temp_b = b;
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
    temp_b = temp_b.array() - b.array();
    float std = sqrt( ( temp_b.array() - temp_b.mean() ).square().sum() / (temp_b.size() - 1) );
    float SNR = mean/std;
    cout << SNR << endl;
}

PYBIND11_MODULE(ctvlib, m)
{
    m.doc() = "C++ Scripts for TV-Tomography Reconstructions";
    py::class_<ctvlib> ctvlib(m, "ctvlib");
    ctvlib.def(py::init<int,int>());
    ctvlib.def("parallelRay", &ctvlib::parallelRay, "Construct Measurement Matrix");
    ctvlib.def("create_measurement_matrix", &ctvlib::loadA, "Load Measurement Matrix Created By Python");
    ctvlib.def("ART", &ctvlib::ART, "ART Tomography");
    ctvlib.def("ART2", &ctvlib::ART2, "Dynamic ART Tomography");
    ctvlib.def("SIRT", &ctvlib::SIRT, "SIRT Tomography");
    ctvlib.def("forwardProjection", &ctvlib::forwardProjection, "Forward Projection");
    ctvlib.def("rowInnerProduct", &ctvlib::normalization, "Calculate the Row Inner Product for Measurement Matrix");
    ctvlib.def("tv_loop", &ctvlib::tv_loop, "TV Gradient Descent Loop");
    ctvlib.def("CosAlpha", &ctvlib::CosAlpha, "Measure Cosine-Alpha");
    ctvlib.def("poissonNoise", &ctvlib::poissonNoise, "Add Poisson Noise to Projections");
}
