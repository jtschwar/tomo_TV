// Minimize the Objects TV with ASD - POCS. 
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
int Niter = 200;

//Number of iterations in TV loop.
int ng = 20;

//Parameter in ART Reconstruction.
float beta = 1.0;

//ART reduction.
float beta_red = 0.995;

//Data Tolerance Parameter
float eps = 1.0;

//dPOCS and reduction criteria
float r_max = 0.95;
float alpha_red = 0.95;
float alpha = 0.2;

// Number of Counts for Poisson Noise. 
int Nc = 100;

///////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    
    //Load Dataset.
    Mat img = imread(filename, cv::ImreadModes::IMREAD_GRAYSCALE);
    int Nslice = img.rows;
    int Nray = img.cols;
    Eigen::MatrixXf tiltSeries;
    cv::cv2eigen(img, tiltSeries);
    
    // Increase the Sampling By 20 degree incremenets.
    for(int k= 0; k < 8; k++)
    {
        
        //Number of Projections for Forward Model.
        int Nproj = 20 + 20 * k;
        
        //Generate Measurement Matrix.
        VectorXf tiltAngles = VectorXf::LinSpaced(Nproj + 1, 0, Nproj);
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
        VectorXf cos_alpha_vec(Niter);

        //Main Loop.
        for(int i=0; i < Niter; i++)
        {
            // Print the Iteration Number.
            if ( i % 100 == 0)
                cout << "Iteration: " << i + 1 << " / " << Niter << "\n";
            
            temp_recon = recon;
            beta_vec(i) = beta;
            
            //ART Reconstruction.
            recon.resize(Ncol, 1);
            tomography2D(recon, b, rowInnerProduct, A, beta);
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

        }
        
        //Save all the vectors.
        saveVecTxt(beta_vec, Nproj, "beta");
        saveVecTxt(dd_vec, Nproj, "dd");
        saveVecTxt(dp_vec, Nproj, "dp");
        saveVecTxt(dg_vec, Nproj, "dg");
        saveVecTxt(dPOCS_vec, Nproj, "dPOCS");
        saveVecTxt(rmse_vec, Nproj, "RMSE");
        saveVecTxt(cos_alpha_vec, Nproj, "Cos_Alpha");
        
        //Save the Image.
        recon.resize(Nslice, Nray);
        Mat final_img;
        cv::eigen2cv(recon, final_img);
        final_img /= recon.maxCoeff();
        imwrite("Results/" + to_string(Nproj) + ".tif", final_img)
        
    }
    
    return 0;
}