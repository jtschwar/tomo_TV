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
    
    int Nray = 20;
    VectorXd offsets = VectorXd::LinSpaced( Nray, -(Nray-1)/2, (Nray-1)/2 );
    VectorXd xrayRoated = (offsets.array() - 1);
    cout << offsets.transpose() << "\n";
    cout << xrayRoated.transpose();
    
    return 0;
}
