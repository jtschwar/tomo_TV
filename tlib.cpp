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

using namespace Eigen;

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

void parallelRay(int& Nray, VectorXf& angles)
{
    //Nside = Nray
    int Nproj = angles.cols(); //Number of projections.
    int idxend = 0;
    
    
    //Ray coordinates at 0 degrees.
    VectorXd offsets = VectorXd::LinSpaced( Nray, -(Nray-1)/2, (Nray-1)/2 );
    VectorXd xgrid = VectorXd::LinSpaced( Nray + 1, - Nray*0.5 , Nray*0.5 );
    VectorXd ygrid = VectorXd::LinSpaced( Nray + 1, - Nray*0.5 , Nray*0.5 );
    
    for(int i=0; i < Nproj; i++)
    {
        float ang = angles(i) / 180;
    }
}

