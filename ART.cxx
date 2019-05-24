// ART Reconstruction. 
// tomo_tv.cxx
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

using namespace Eigen;
using namespace std;
using namespace cv;

///////////////RECONSTRUCTION PARAMETERS///////////////////

//File Name (Input Tilt Series).
String filename = "phantom.tif";

//Total Number of Iterations.
int Niter = 100;

//Number of Projections for Forward Model.
int Nproj = 30;

//Parameter in ART Reconstruction.
float beta = 1.0;

//ART reduction.
float beta_red = 0.995;

///////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    
//    int n = 5;
//    Eigen::setNbThreads(n);
    
    //Load Dataset.
    Mat img = imread("Test_Images/" + filename, cv::ImreadModes::IMREAD_GRAYSCALE);
    int Nslice = img.rows;
    int Nray = img.cols;
    Eigen::MatrixXf tiltSeries;
    cv::cv2eigen(img, tiltSeries);

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

    MatrixXf recon (Nslice, Nray);
    recon.setZero();

    //Main Loop.
    for(int i=0; i < Niter; i++)
    {
        if ( i % 10 == 0)
            cout << "Iteration: " << i + 1 << " / " << Niter << "\n";

        //ART Reconstruction.
        tomography(recon, b, rowInnerProduct, A, beta);
        recon = (recon.array() < 0).select(0, recon);
        beta *= beta_red;
    }

//    Display and Save final reconstruction.
    // recon.resize(Nslice, Nray);
    Mat final_img;
    cv::eigen2cv(recon, final_img);
    final_img /= recon.maxCoeff();

    namedWindow( "Reconstruction", WINDOW_AUTOSIZE );
    imshow( "Reconstruction", final_img );
    waitKey(0);

    return 0;
}
