// Minimize the Objects TV with ASD - POCS.
// tomo_tv.cxx
//  TV
//
//  Created by Jonathan Schwartz on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include <iostream>
#include <sys/stat.h>
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
String filename = "Co2P_v2.tif";

int Niter, ng;                 //Number of Iterations (Main and TV).
float dTheta;                  //Step Size for Theta.
float beta, beta_red;          //Parameter in ART Reconstruction.
float eps;                     //Data Tolerance Parameter.
float alpha, alpha_red, r_max; //TV Parameter and reduction criteria.

int Nc = 100; // Number of Counts for Poisson Noise. 

///////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    
    //Assign Values from parameters.txt
    read_parameters(Niter, ng, dTheta, beta, beta_red, alpha, alpha_red, eps, r_max);
    
    //Load Dataset.
    Mat img = imread("Test_Images/" + filename, cv::ImreadModes::IMREAD_GRAYSCALE);
    int Nslice = img.rows;
    int Nray = img.cols;
    Eigen::MatrixXf tiltSeries;
    cv::cv2eigen(img, tiltSeries);

    //Generate Measurement Matrix.
    int Nproj = 180/dTheta + 1;
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
    //poissonNoise(b, Nc); //Uncoment if you'd like to add poisson Noise.
    tiltSeries.resize(Nslice, Nray);
    
    // Empty Vectors/Matrices for Reconstruction. 
    VectorXf g;
    MatrixXf recon (Nslice, Nray), temp_recon(Nslice, Nray), v;
    recon.setZero();
    float dPOCS;
    
    //Vectors to evalutate convergence.
    VectorXf dd_vec(Niter), dp_vec(Niter), dg_vec(Niter);
    VectorXf dPOCS_vec(Niter), beta_vec(Niter), rmse_vec(Niter);
    VectorXf cos_alpha_vec(Niter), tv_vec(Niter);

    //Main Loop.
    for(int i=0; i < Niter; i++)
    {
        // Print the Iteration Number.
        if ( i % 100 == 0)
            cout << "Iteration: " << i + 1 << " / " << Niter << "\n";
        
        temp_recon = recon;
        beta_vec(i) = beta;
        
        //ART Reconstruction.
        tomography(recon, b, rowInnerProduct, A, beta);
        recon = (recon.array() < 0).select(0, recon);
        g = A * recon;
        recon.resize(Nslice, Nray);
        
        if(i == 0)
        {
            dPOCS = (recon - temp_recon).norm() * alpha;
        }
        
        dd_vec(i) = (g - b).norm() / g.size();
        dp_vec(i) = (temp_recon - recon).norm();
        temp_recon = recon;

        //TV Loop.
        for(int j=0; j<ng; j++)
        {
            v = tv2Dderivative(recon);
            v /= v.norm();
            recon.array() -= dPOCS * v.array();
        }

        dg_vec(i) = (recon - temp_recon).norm();

        //Reduce TV if data constraint isn't met.
        if (dg_vec(i) > dp_vec(i) * r_max && dd_vec(i) > eps)
        {
            dPOCS *= alpha_red;
        }
        
        dPOCS_vec(i) = dPOCS;
        beta *= beta_red;
        rmse_vec(i) = (tiltSeries - recon).norm();
        cos_alpha_vec(i) = CosAlpha(recon, v, g, b, A);
        tv_vec(i) = tv2D(recon);

    }

    //Create Directory to Save Results.
    mkdir("Results/ASD_tv", ACCESSPERMS);
    
    //Save all the vectors.
    saveVec(beta_vec, "beta");
    saveVec(dd_vec, "dd");
    saveVec(dp_vec, "dp");
    saveVec(dg_vec, "dg");
    saveVec(dPOCS_vec, "dPOCS");
    saveVec(rmse_vec, "RMSE");
    saveVec(cos_alpha_vec, "Cos_Alpha");
    saveVec(tv_vec, "TV");

    //Display the final reconstruction.
//    Mat final_img;
//    cv::eigen2cv(recon, final_img);
//    namedWindow( "Reconstruction", WINDOW_AUTOSIZE );
//    imshow( "Reconstruction", final_img * (1.0 / recon.maxCoeff()) );
//    waitKey(0);
    
    //Save the Image.
    recon.resize(Nslice, Nray);
    Mat final_img;
    cv::eigen2cv(recon, final_img);
    final_img /= recon.maxCoeff();
    imwrite("Results/ASD_tv/recon.tif", final_img);
    
    return 0;
}
