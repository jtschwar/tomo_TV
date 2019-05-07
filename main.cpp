//
//  main.cpp
//  TV
//
//  Created by Jonathan Schwartz on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int Niter = 100;
int ng = 20;
int beta = 1.0;
int beta_red = 095;
int gamma_red;

int main(int argc, const char * argv[]) {
    Matrix3f A;
    Vector3f b;
    A << 1,2,3,  4,5,6,  7,8,10;
    b << 3,3,4;
    
    cout << "Here is Matrix A" << A << endl;
    cout << "Here is Vector b" << b << endl;
    
    Vector3f x = A.colPivHouseholderQr().solve(b);
    cout << "The solution is: " << x << endl;
    
    return 0;
}
