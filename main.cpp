//
//  main.cpp
//  TV
//
//  Created by Jonathan Schwartz on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include <iostream>
#include <Eigen/Core>
#include <numeric>
#include "tlib.cpp"

using namespace Eigen;
using namespace std;

int Niter = 100;
int ng = 20;
int beta = 1.0;
float beta_red = 0.95;
int gamma_red = -5;

int main(int argc, const char * argv[]) {
    
    VectorXd offsets(10);
    offsets(0) = 1;
    offsets(1) = 2;
    cout << offsets.transpose() << "\n\n";
    //offsets.resize(5), offsets.setZero();
    //cout << offsets.transpose() << "\n";
   // cout << xrayRoated.transpose();
    
    return 0;
}
