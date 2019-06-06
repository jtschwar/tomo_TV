//  Simulate tomography reconstructions when a new projection
//  is added every three minutes. 
//
//  Created by Jonathan Schwartz on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include <iostream>
#include <time.h>
#include <sys/stat.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "./Utils/dynamic_tlib.cpp"

using namespace Eigen;
using namespace std;
using namespace cv;

///////////////RECONSTRUCTION PARAMETERS///////////////////

//File Name (Input Tilt Series).
String filename = "Co2P_256.tif";

int theta_block = 1;

int Niter, ng;                 //Number of Iterations (Main and TV).
float Niter_red;               // Number of Iterations Reduction.
float dTheta;                  //Step Size for Theta.
float beta0, beta_red;          //Parameter in ART Reconstruction.
float eps;                     //Data Tolerance Parameter.
float alpha, alpha_red, r_max; //TV Parameter and reduction criteria.

///////////////////////////////////////////////////////////

//Time .
float timer;              // Timer = track time elapsed.
int i;                    // Track number of iterations completed per recon.
clock_t t0;               // Clock.
float time_limit = 180.0 / 512.0 * 8.0; // Total Time to Run Reconstruction (s).
//float Ncores = 16;        // Ncores to simulate.

int main(int argc, const char * argv[]) {
    
    //Assign Values from parameters.txt
    read_parameters(Niter, Niter_red, ng, dTheta, beta0, beta_red, alpha, alpha_red, eps, r_max);  

    //Load Dataset.
    Mat img = imread("Test_Images/" + filename, cv::ImreadModes::IMREAD_GRAYSCALE);
    int Nslice = img.rows;
    int Nray = img.cols;
    Eigen::MatrixXf tiltSeries;
    cv::cv2eigen(img, tiltSeries);
    
    // Empty Vectors/Matrices for Reconstruction.
    VectorXf g;
    MatrixXf recon (Nslice, Nray), temp_recon(Nslice, Nray), v;
    recon.setZero();
    
    VectorXf Niter_vec(180);
    float dp, dg, dPOCS, beta;
    int theta_max, Nproj, Nrow, Ncol;
    
    //Number of Projections for Forward Model.
    theta_max = 180;
    Nproj = theta_max/dTheta + 1;
    
    //Generate Measurement Matrix.
    VectorXf tiltAngles = VectorXf::LinSpaced(Nproj, 0, theta_max );
    Nrow = Nray * Nproj;          // Number of Rows in Measurement Matrix (A)
    Ncol = Nray * Nray;           // Number of Columns in Measurement Matrix (A)
    
    SparseMatrix<float, Eigen::RowMajor> A(Nrow,Ncol);
    parallelRay(Nray, tiltAngles, A);

    //Calculate Inner Product.
    VectorXf rowInnerProduct(Nrow);
    for(int j=0; j < Nrow; j++)
    {
        rowInnerProduct(j) = A.row(j).dot(A.row(j));
    }
    
    // Create Projections.
    tiltSeries.resize(tiltSeries.size(), 1);
    VectorXf b = A * tiltSeries;
    tiltSeries.resize(Nslice, Nray);
    
    theta_max = 1;
    
    // Increase the Sampling By 1 Degree for 20 degree chunks.
    for(int k= 0; k < 180; k++)
    {
        cout << "\nReconstructing  Tilt Angles: 0 -> " << theta_max << " Degrees" <<endl;
        
        //Parameter in ART Reconstruction.
        beta = beta0;

        //Vectors to evalutate convergence.
        VectorXf dd_vec(Niter), rmse_vec(Niter), tv_vec(Niter);
        dd_vec.setZero(), rmse_vec.setZero(), tv_vec.setZero();
        Nrow = Nray * theta_max;

        i = 0;
        t0 = clock();
        
        //Main Loop.
        do {
            
            temp_recon = recon;

            //ART Reconstruction.
            ART(recon, b, rowInnerProduct, A, beta, Nrow);
            recon = (recon.array() < 0).select(0, recon);
            g = A.topRows(Nrow) * recon;
            recon.resize(Nslice, Nray);

            if(k == 0 && i == 0)
            {
                dPOCS = (recon - temp_recon).norm() * alpha;
            }

            dd_vec(i) = (g - b.topRows(Nrow)).norm() / g.size();
            dp = (temp_recon - recon).norm();
            temp_recon = recon;

            //TV Loop.
            for(int j=0; j<ng; j++)
            {
                v = tv2Dderivative(recon);
                v /= v.norm();
                recon.array() -= dPOCS * v.array();
            }

            dg = (recon - temp_recon).norm();

            //Reduce TV if data constraint isn't met.
            if (dg > dp * r_max && dd_vec(i) > eps)
            {
                dPOCS *= alpha_red;
            }

            beta *= beta_red;
            rmse_vec(i) = (tiltSeries - recon).norm();
            tv_vec(i) = tv2D(recon);
            
            ++i;
            timer = (clock() - t0)/1e6;
            
        } while (timer < time_limit);
        
        Niter_vec(k) = i;
        cout << "Number of Iterations: " << i << endl;
        
        //Create Directory to Save Results.
        String directory = "Results/" + to_string(theta_max);
        mkdir(directory.c_str(), ACCESSPERMS);
        
        //Save all the vectors.
        saveResults(dd_vec, theta_max, "dd");
        saveResults(rmse_vec, theta_max, "RMSE");
        saveResults(tv_vec, theta_max, "TV");
        
        if (k == 179)
            saveResults(Niter_vec, theta_max, "Niter");
        
        //Save the Image.
        recon.resize(Nslice, Nray);
        Mat final_img;
        cv::eigen2cv(recon, final_img);
        final_img /= recon.maxCoeff();
        imwrite("Results/" + to_string(theta_max) + "/recon.tif", final_img);
        
        if (Niter > 1.0)
            Niter *= Niter_red;
        
        theta_max++;
        
    }
    
    return 0;
}
