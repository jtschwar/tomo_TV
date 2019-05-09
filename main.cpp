//
//  main.cpp
//  TV
//
//  Created by Jonathan Schwartz on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include "tlib.cpp"

//using namespace Eigen;
using namespace std;

//Reconstruction parameters.
int Niter = 25;
int beta = 1.0;
float beta_red = 0.95;

int main(int argc, const char * argv[]) {
    
    string fileName = "phantom.tif";
    ifstream inputData;
    
    //cout << img;
    //img = imread("phantom.tif");
    //SparseMatrix<float, RowMajor> A;
    
    return 0;
}
