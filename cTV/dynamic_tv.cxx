// Simulate Dynamic TV when data comes in chunks of projections.
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
#include "./Utils/dynamic_tlib.cpp"

using namespace Eigen;
using namespace std;
using namespace cv;

///////////////RECONSTRUCTION PARAMETERS///////////////////

//File Name (Input Tilt Series).
String filename = "Co2P_256.tif";

int Niter, ng;                 //Number of Iterations (Main and TV).
float Niter_red;               // Number of Iterations Reduction.
float dTheta;                  //Step Size for Theta.
float beta0, beta_red;          //Parameter in ART Reconstruction.
float eps;                     //Data Tolerance Parameter.
float alpha, alpha_red, r_max; //TV Parameter and reduction criteria.

// Number of Counts for Poisson Noise. 
int Nc = 100;

///////////////////////////////////////////////////////////

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
    float dPOCS;

    //Number of Projections for Forward Model.
    int theta_max = 180;
    int Nproj = theta_max/dTheta + 1;
    
    //Generate Measurement Matrix.
    VectorXf tiltAngles = VectorXf::LinSpaced(Nproj, 0, theta_max );
    int Nrow = Nray * Nproj;
    int Ncol = Nray * Nray;
    
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
    
    // Increase the Sampling By 1 Degree for 20 degree chunks.
    for(int k= 0; k < 9; k++)
    {
        //Parameter in ART Reconstruction.
        float beta = beta0;
        
        theta_max = 20 + 20 * k;
        Nproj = theta_max/dTheta + 1;
        Nrow = Nray * Nproj;

        cout << "\nReconstructing  Tilt Angles: 0 -> " << theta_max << " Degrees" <<endl;
        
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
            ART(recon, b, rowInnerProduct, A, beta, Nrow);
            recon = (recon.array() < 0).select(0, recon);
            g = A.topRows(Nrow) * recon;
            recon.resize(Nslice, Nray);
            
            if(k == 0 && i == 0)
            {
                dPOCS = (recon - temp_recon).norm() * alpha;
            }
            
            dd_vec(i) = (g - b.topRows(Nrow)).norm() / g.size();
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
            rmse_vec(i) = sqrt(((tiltSeries - recon).array().square()).mean());
            // cos_alpha_vec(i) = CosAlpha(recon, v, g, b, A);
            tv_vec(i) = tv2D(recon);

        }
        
        //Create Directory to Save Results.
        String directory = "Results/" + to_string(theta_max);
        mkdir(directory.c_str(), ACCESSPERMS);
        
        //Save all the vectors.
        saveResults(dd_vec, theta_max, "dd");
        saveResults(rmse_vec, theta_max, "RMSE");
        saveResults(tv_vec, theta_max, "TV");
        // saveResults(beta_vec, theta_max, "beta");
        // saveResults(dp_vec, theta_max, "dp");
        // saveResults(dg_vec, theta_max, "dg");
        // saveResults(dPOCS_vec, theta_max, "dPOCS");
        // saveResults(cos_alpha_vec, theta_max, "Cos_Alpha");
       
        
        //Save the Image.
        recon.resize(Nslice, Nray);
        Mat final_img;
        cv::eigen2cv(recon, final_img);
        final_img /= recon.maxCoeff();
        imwrite("Results/" + to_string(theta_max) + "/recon.tif", final_img);
        
        Niter *= Niter_red; //Niter_red.
        
    }
    
    return 0;
}
