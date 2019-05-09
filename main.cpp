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

using namespace Eigen;
using namespace std;

int Niter = 100;
int ng = 20;
int beta = 1.0;
float beta_red = 0.95;
int gamma_red = -5;

int main(int argc, const char * argv[]) {
    
    Vector2d row, col, val;
    row(0) = 0;
    row(1) = 1;
    col(0) = 0;
    col(1) = 0;
    val(0) = 10;
    val(1) = 20;
    SparseMatrix<double, RowMajor> sm(3,3);
    
    for(int i = 0; i < 2; i++)
    {
        sm.insert(row(i), col(i)) = val(i);
    }
    
    cout << sm << '\n';
    
    return 0;
}
