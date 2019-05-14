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
int Niter = 25;
int beta = 1.0;
float beta_red = 0.95;
int Nproj = 30;
String filename = "phantom.tif";

int main(int argc, const char * argv[]) {
    
    // Load Dataset. 
    Mat img = imread(filename, cv::ImreadModes::IMREAD_GRAYSCALE);
    int Nslice = img.rows;
    int Nray = img.cols;
    Eigen::MatrixXf tiltSeries;
    cv::cv2eigen(img, tiltSeries);
    
    //Display Original Image.
//    namedWindow( "Original Image", WINDOW_AUTOSIZE );
//    imshow( "Original Image", img );
//    waitKey(0);

    //Generate Measurement Matrix.
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
    tiltSeries.resize(tiltSeries.size(), 1);
    VectorXf b = A * tiltSeries;
    VectorXf vec_recon(Ncol,1);
    vec_recon.setZero();

    //Main Loop.
    for(int i=0; i < Niter; i++)
    {
        cout << "Iteration: " << i + 1 << " / " << Niter << "\n";
        tomography2D(vec_recon, b, rowInnerProduct, A, beta);
        vec_recon = (vec_recon.array() < 0).select(0, vec_recon);
    }

    //Display and Save final reconstruction.
    MatrixXf recon;
    Map<MatrixXf> temp_recon(vec_recon.data(), Nray, Nray);
    recon = temp_recon;
    Mat final_img;
    cv::eigen2cv(recon, final_img);

    namedWindow( "Reconstruction", WINDOW_AUTOSIZE );
    imshow( "Reconstruction", final_img * (1.0 / 255) );
    waitKey(0);
    
    return 0;
}
