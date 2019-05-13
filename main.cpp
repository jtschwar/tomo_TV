//
//  main.cpp
//  TV
//
//  Created by Jonathan Schwartz on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "tlib.cpp"

//using namespace Eigen;
using namespace std;

//Reconstruction parameters.
int Niter = 25;
int beta = 1.0;
float beta_red = 0.95;
int Nslice = 256;
int Nray = 256;
int Nproj = 30;
int Nrow = Nray * Nproj;
int Ncol = Nray * Nray;


int main(int argc, const char * argv[]) {
    
    VectorXf tiltAngles = VectorXf::LinSpaced(Nproj, 0, 180);
    SparseMatrix<float, Eigen::RowMajor> A(Nrow, Ncol);
    parallelRay(Nray, tiltAngles, A);
    
    VectorXf rowInnerProduct(Nrow);
    for(int j=0; j < Nrow; j++)
    {
        rowInnerProduct(j) = A.row(j).dot(A.row(j));
    }
    
    return 0;
}
