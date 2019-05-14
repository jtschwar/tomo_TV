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
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "tlib.cpp"

#define EIGEN_RUNTIME_NO_MALLOC

using namespace Eigen;
using namespace std;
using namespace cv;

//Reconstruction parameters.
int Niter = 5;
int beta = 1.0;
float beta_red = 0.95;
String filename = "phantom.tif";

int main(int argc, const char * argv[]) {
    
    Mat img = imread(filename);
    int Nslice = img.rows;
    int Nray = img.cols;
    int Nproj = 30;
    
    //Generate Measurement Matrix.
    Map<MatrixXf> img_matrix(img.ptr<float>(), Nslice, Nray);
    cout << img_matrix.rows() ;
    
    VectorXf tiltAngles = VectorXf::LinSpaced(Nproj, 0, 180);
    int Nrow = Nray*Nproj;
    int Ncol = Nray*Nray;
    SparseMatrix<float, Eigen::RowMajor> A(Nrow,Ncol);
    parallelRay(Nray, tiltAngles, A);
    
    //Calculate Inner Product.
    VectorXf rowInnerProduct(Nrow);
    for(int j=0; j < Nrow; j++)
    {
        rowInnerProduct(j) = A.row(j).dot(A.row(j));
    }
    
    //Vectorize/Initialize the reconstruction and experimental data.
    Map<VectorXf> tiltSeries(img_matrix.data(), img_matrix.size());
    VectorXf b = A * tiltSeries;
    VectorXf recon(tiltSeries.size());
    recon.setZero();
    
    //Main Loop. 
    for(int i=0; i < Niter; i++)
    {
        cout << "Iteration: " << i + 1 << " / " << Niter << "\n";
        tomography2D(recon, b, rowInnerProduct, A, beta);
    }
    
    //Display and Save final reconstruction.
    Map<MatrixXf> finalRecon(recon.data(), Nslice, Nray);
    Mat final_img(Nray, Nray, CV_32FC1, finalRecon.data());
    
    //cout << final_img;

    //namedWindow( "Reconstruction", WINDOW_AUTOSIZE );
    //imshow( "Reconstruction", final_img );
    //waitKey(0);
    
    return 0;
}
