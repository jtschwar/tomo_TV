//
//  tlib.cpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "tlib.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <iostream>

#define PI 3.14159265

using namespace Eigen;
using namespace std;

void tomography(Eigen::MatrixXf& recon, Eigen::MatrixXf& tiltSeries, Eigen::VectorXf& innerProduct, Eigen::SparseMatrix<float>& A, int beta)
{
    Map<RowVectorXf> b(tiltSeries.data(), tiltSeries.size());
    Map<RowVectorXf> f(recon.data(), recon.size());
    
    long Nrow = A.rows();
    long Nray = recon.rows();
    float a;
    
    for(int j=0; j < Nrow; j++)
    {
        a = (b(j) - A.row(j).dot(f)) / innerProduct(j);
        f = f + A.row(j) * a * beta;
    }
    f.resize(Nray, Nray);
    recon = f;
}

float rmepsilonScalar(float input)
{
    if (abs(input) < 1e-10)
        input = 0;
    return input;
}

void rmepsilonVector(Eigen::VectorXf& input)
{
    input = (input.array().abs() < 1e-8).select(0, input);
}

void parallelRay(int Nray, Eigen::VectorXf angles)
{
    //Nside = Nray = y dimension of tilt series.
    
    int Nproj = angles.cols(); //Number of projections.
    //int idxend = 0;
    
    //Initialize vectors that contain matrix elements and corresponding row/column numbers.
    //Ray coordinates at 0 degrees.
    VectorXd offsets = VectorXd::LinSpaced( Nray, -(Nray-1)/2, (Nray-1)/2 );
    VectorXd xgrid = VectorXd::LinSpaced( Nray + 1, - Nray*0.5 , Nray*0.5 );
    VectorXd ygrid = VectorXd::LinSpaced( Nray + 1, - Nray*0.5 , Nray*0.5 );
    
    //Loop over projection angles.
    for(int i=0; i < Nproj; i++)
    {
        float ang = angles(i) * PI / 180;
        //Points passed by rays at current angles.
        VectorXd xrayRoated = cos(ang) * offsets;
        VectorXd yrayRoated = sin(ang) * offsets;
        xrayRoated = (xrayRoated.array().abs() < 1e-8).select(0, xrayRoated);
        yrayRoated = (yrayRoated.array().abs() < 1e-8).select(0, yrayRoated);
        
        float a = -sin(ang);
        a = rmepsilonScalar(a);
        float b = cos(ang);
        b = rmepsilonScalar(b);
        
        //Loop rays in current projection.
        for(int j = 0; j < Nray; j++ )
        {
            //Ray: y = tx * x + intercept.
            VectorXd t_xgrid, y_xgrid, t_ygrid, x_ygrid;
            t_xgrid = (xgrid.array() - xrayRoated(j)) / a;
            y_xgrid = b * t_xgrid.array() + yrayRoated(j);
            t_ygrid = (ygrid.array() - yrayRoated(j)) / b;
            x_ygrid = a * t_ygrid.array() + xrayRoated(j);
            
            //Collect all points
            long tne = t_xgrid.size() + t_ygrid.size(); // tne = total number of elements
            VectorXd t_grid(tne), xx_temp(tne), yy_temp(tne);
            t_grid << t_xgrid, t_ygrid;
            xx_temp << xgrid, x_ygrid;
            yy_temp << y_xgrid, ygrid;
            
            //Sort the coordinates according to intersection time.
            VectorXd I = VectorXd::LinSpaced(t_grid.size(), 0, t_grid.size()-1);
            std::sort(I.data(), I.data() + I.size(), [&t_grid] (size_t i1, size_t i2) {return t_grid[i1] < t_grid[i2];} );
            VectorXd xx(I.size()), yy(I.size());
            for(int k=0; k<t_grid.size(); k++){
                xx(k) = xx_temp(I(k));
                yy(k) = yy_temp(I(k));
            }

            // Get rid of points that are outside the image grid.
        
        }
        
    }
}
