//
//  main.cpp
//  TV
//
//  Created by Jonathan Schwartz on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include <iostream>
#include <Eigen/Dense>
#include "tlib.cpp"

//using namespace Eigen;
using namespace std;

int Niter = 100;
int ng = 20;
int beta = 1.0;
float beta_red = 0.95;
int gamma_red = -5;

int main(int argc, const char * argv[]) {
    
    beta_red = rmepsilon(gamma_red);
    return 0;
}
