// Minimize the Objects TV with a Projected Line Search. 
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

//Number of iterations in TV loop.
int ng = 20;

//Parameter in ART Reconstruction.
float beta = 1.0;

//ART reduction.
float beta_red = 0.995;

//TV Reduction.
float gamma_red = 0.8;

///////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    
    //Load Dataset.
    Mat img = imread(filename, cv::ImreadModes::IMREAD_GRAYSCALE);
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
    MatrixXf recon (Nslice, Nray), temp_recon(Nslice, Nray), v, recon_prime;;
    float dPOCS,R0, Rf;

    //Main Loop.
    for(int i=0; i < Niter; i++)
    {
        if (i % 10 == 0)
            cout << "Iteration: " << i + 1 << " / " << Niter << "\n";
        temp_recon = recon;

        //ART Reconstruction.
        tomography(recon, b, rowInnerProduct, A, beta);
        recon.resize(Nslice, Nray);
        recon = (recon.array() < 0).select(0, recon);
        beta *= beta_red;

        if (i == 0)
            dPOCS = (recon - temp_recon).norm();
        float dp = (temp_recon - recon).norm();
        temp_recon = recon;

        for(int j=0; j<ng; j++)
        {
            R0 = tv2D(recon);
            v = tv2Dderivative(recon);
            recon_prime = recon.array() - dPOCS * v.array();
            recon_prime = (recon_prime.array() < 0).select(0, recon_prime);
            Rf = tv2D(recon_prime);
            float gamma = 1.0;

            while (Rf > R0)
            {
                gamma *= gamma_red;
                recon_prime = recon.array() - gamma * dPOCS * v.array();
                Rf = tv2D(recon_prime);
                recon_prime = (recon_prime.array() < 0).select(0, recon_prime);
            }
            recon = recon_prime;
        }

        float dg = (recon - temp_recon).norm();

        if (dg > dp)
        {
            dPOCS *= 0.8;
        }
    }

    //Display and Save final reconstruction.
    Mat final_img;
    cv::eigen2cv(recon, final_img);

    namedWindow( "Reconstruction", WINDOW_AUTOSIZE );
    imshow( "Reconstruction", final_img * (1.0 / 255) );
    waitKey(0);
    
    return 0;
}
