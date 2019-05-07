//
//  tlib.cpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "tlib.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

#define PI 3.14159265

using namespace Eigen;
using namespace std;

void tomography(MatrixXf& recon, MatrixXf& tiltSeries, SparseMatrix<float>& A, int beta)
{
    Map<RowVectorXf> b(tiltSeries.data(), tiltSeries.size());
    Map<RowVectorXf> f(recon.data(), recon.size());
    
    int Nrow = A.rows();
    int Nray = recon.rows();
    float a;
    
    for(int j=0; j < Nrow; j++)
    {
        a = (b(j) - A.row(j).dot(f)) / (A.row(j).dot(A.row(j)));
        f = f + A.row(j) * a * beta;
    }
    f.resize(Nray, Nray);
    recon = f;
}

float rmepsilon(float input)
{
    if (abs(input) < 1e-10)
        input = 0;
    return input;
}

void parallelRay(int& Nray, VectorXf& angles)
{
    //Nside = Nray
    int Nproj = angles.cols(); //Number of projections.
    int idxend = 0;
    
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
        xrayRoated = (xrayRoated.array() < 1e-8).select(0, xrayRoated);
        yrayRoated = (yrayRoated.array() < 1e-8).select(0, yrayRoated);
        
        float a = -sin(ang);
        a = rmepsilon(a);
        float b = cos(ang);
        b = rmepsilon(b);
        
        //Loop rays in current projection.
        for(int j = 0; j < Nray; j++ )
        { //Ray: y = tx * x + intercept.
            VectorXd t_xgrid, y_xgrid;
            t_xgrid = (xgrid - xrayRoated(j)) / a;
            VectorXd y_xgrid = b * t_xgrid + yrayRoated(j);
            
            VectorXd t_ygrid = (ygrid - yrayRoated(j)) / b;
            VectorXd x_ygrid = a * t_ygrid + xrayRoated(j);
            
            //Collect all points
            VectorXd t_grid, xx, yy;
            t_grid << t_xgrid, t_ygrid;
            xx << xgrid, x_ygrid;
            yy << y_xgrid, ygrid;
            
            //Sort the coordinates according to intersection time.
            
            
            // Get rid of points that are outside the image grid.
        
        }
        
    }
}
