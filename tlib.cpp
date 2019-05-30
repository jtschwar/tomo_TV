//
//  tlib.cpp
//  TV
//
//  Created by Hovden Group on 5/6/19.
//  Copyright © 2019 Jonathan Schwartz. All rights reserved.
//

#include "tlib.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <iostream>
#include <fstream>
#include <random>

#define PI 3.14159265359

using namespace Eigen;
using namespace std;

void tomography(Eigen::MatrixXf& recon, Eigen::VectorXf& b, Eigen::VectorXf& innerProduct, Eigen::SparseMatrix<float, RowMajor>& A, float beta)
{
    //2D ART Tomography
    long Nrow = A.rows();
    long Ncol = A.cols();
    long Nx = recon.rows();
    long Ny = recon.cols();
    float a;
    
    recon.resize(Ncol, 1);
    VectorXf f(recon.rows());
    f = recon;
    
    for(int j=0; j < Nrow; j++)
    {
        a = (b(j) - A.row(j).dot(f)) / innerProduct(j);
        f += A.row(j).transpose() * a * beta;
    }
    recon = f;
}

float CosAlpha(Eigen::MatrixXf& recon, Eigen::MatrixXf& tv_derivative, Eigen::VectorXf& g, Eigen::VectorXf& b, Eigen::SparseMatrix<float, Eigen::RowMajor>& A)
{
    float cosA, Nx, Ny, norm;
    int Ncol;

    Nx = recon.rows();
    Ny = recon.cols();
    Ncol = A.cols();

    MatrixXf d_tv(Ncol,1), d_data(Ncol,1);
    d_tv.setZero(), d_data.setZero();

    //Vectorize.
    recon.resize(Ncol,1), tv_derivative.resize(Ncol,1);

    VectorXf nabla_h(Ncol);
    nabla_h = 2 * A.transpose() * ( g - b);

    for(int i=0; i < recon.size(); i++ )
    {
        if( abs(recon(i)) > 1e-10 )
        {
            d_tv(i) = tv_derivative(i);
            d_data(i) = nabla_h(i);
            cosA += d_data(i) * d_tv(i);
        }
    }
    recon.resize(Nx, Ny);
    norm = d_data.norm() * d_tv.norm();
    cosA /= norm;
    return cosA;
}

Eigen::MatrixXf tv2Dderivative(Eigen::MatrixXf recon)
{
    int padX = recon.rows() + 2;
    int padY = recon.cols() + 2;
    
    MatrixXf v1n, v1d, v2n, v2d, v3n, v3d, v;
    MatrixXf r(padX, padY), rXp(padX, padY), rYp(padX, padY);
    r.setZero(), rXp.setZero(), rYp.setZero();
    circshift(recon, r, 0, 0), circshift(recon, rXp, 1, 0), circshift(recon, rYp, 0, 1);
    v1n = 4 * r.array() - 2 * rXp.array() - 2 * rYp.array();
    v1d = sqrt( 1e-8 + (r - rXp).array().square() + (r - rYp).array().square() );
    rXp.resize(0,0), rYp.resize(0,0);
    
    MatrixXf rXn(padX, padY), rXnYp(padX, padY);
    rXn.setZero(), rXnYp.setZero();
    circshift(recon, rXn, -1, 0), circshift(recon, rXnYp, -1, 1);
    v2n = 2 * (r - rXn).array();
    v2d = sqrt( 1e-8 + (rXn - r).array().square() + (rXn - rXnYp).array().square() );
    rXn.resize(0,0), rYp.resize(0,0);
    
    MatrixXf rYn(padX, padY), rXpYn(padX, padY);
    rYn.setZero(), rXpYn.setZero();
    circshift(recon, rYn, 0, -1), circshift(recon, rXpYn, 1, -1);
    v3n = 2 * (r - rYn).array();
    v3d = sqrt( 1e-8 + (rYn - r).array().square() + (rYn - rXpYn).array().square() );
    rYn.resize(0,0), rXpYn.resize(0,0);
    
    v = v1n.array() / v1d.array() + v2n.array() / v2d.array() + v3n.array() / v3d.array();
    MatrixXf v2(recon.rows(), recon.cols());
    v2 = v.block(1, 1, recon.rows(), recon.cols());
    v.resize(0,0), v1n.resize(0,0), v1d.resize(0,0), v2n.resize(0,0), v2d.resize(0,0), v3n.resize(0,0), v3d.resize(0,0);
    return v2;
}

float tv2D(Eigen::MatrixXf& recon)
{
    int padX = recon.rows() + 2;
    int padY = recon.cols() + 2;
    MatrixXf temp_tv(recon.rows(), recon.cols());
    temp_tv.setZero();
    float tv;
    MatrixXf r(padX, padY), rXp(padX, padY), rYp(padX, padY);
    r.setZero(), rXp.setZero(), rYp.setZero();
    circshift(recon, r, 0, 0);
    circshift(recon, rXp, 1, 0);
    circshift(recon, rYp, 0, 1);
    MatrixXf mat_tv = sqrt( (r - rXp).array().square() + (r - rYp).array().square() );
    temp_tv = mat_tv.block(1, 1, recon.rows(), recon.cols());
    tv = temp_tv.sum();
    return tv;
}

void circshift(Eigen::MatrixXf input, Eigen::MatrixXf& output, int i, int j)
{
    // i == shift in the x - direction.
    // j == shift in the y - direction.
    output.block(1+i, 1+j, input.rows(), input.rows()) = input;
}

void parallelRay(int& Nray, Eigen::VectorXf& angles, Eigen::SparseMatrix<float, Eigen::RowMajor>& A)
{
    //Nside = Nray = y dimension of tilt series.
    int Nside = Nray;
    
    //When is this not equal to 1??
    float pixelWidth = 1;
    float rayWidth = 1;
    
    //Number of projections.
    int Nproj = angles.rows();
    int idxend = 0;
    
    //Initialize vectors that contain matrix elements and corresponding row/column numbers.
    //Ray coordinates at 0 degrees.
    VectorXf offsets = VectorXf::LinSpaced( Nray + 1, -(Nray - 1) * 0.5, (Nray + 1) * 0.5 ) * rayWidth;
    VectorXf xgrid = VectorXf::LinSpaced( Nray + 1, -Nray * 0.5 , Nray * 0.5) * pixelWidth;
    VectorXf ygrid = VectorXf::LinSpaced( Nray + 1, - Nray * 0.5 , Nray * 0.5 ) * pixelWidth;
    
    //Initialize vectors that contain matrix elements and corresponding row/column numbers
    long int max_elements = 2 * Nside * Nproj * Nray;
    VectorXf rows(max_elements), cols(max_elements), vals(max_elements);
    
    //Loop over projection angles.
    for(int i=0; i < Nproj; i++)
    {
        float ang = angles(i) * PI / 180;
        //Points passed by rays at current angles.
        VectorXf xrayRoated = cos(ang) * offsets;
        VectorXf yrayRoated = sin(ang) * offsets;
        xrayRoated = (xrayRoated.array().abs() < 1e-8).select(0, xrayRoated);
        yrayRoated = (yrayRoated.array().abs() < 1e-8).select(0, yrayRoated);
        
        float a = -sin(ang);
        a = rmepsilon(a);
        float b = cos(ang);
        b = rmepsilon(b);
        
        //Loop rays in current projection.
        for(int j = 0; j < Nray; j++ )
        {
            //Ray: y = tx * x + intercept.
            VectorXf t_xgrid, y_xgrid, t_ygrid, x_ygrid;
            t_xgrid = (xgrid.array() - xrayRoated(j)) / a;
            y_xgrid = b * t_xgrid.array() + yrayRoated(j);
            t_ygrid = (ygrid.array() - yrayRoated(j)) / b;
            x_ygrid = a * t_ygrid.array() + xrayRoated(j);
            
            //Collect all points
            long tne = t_xgrid.size() + t_ygrid.size(); // tne = total number of elements
            VectorXf t_grid(tne), xx_temp(tne), yy_temp(tne);
            t_grid << t_xgrid, t_ygrid;
            xx_temp << xgrid, x_ygrid;
            yy_temp << y_xgrid, ygrid;
            
            //Sort the coordinates according to intersection time.
            VectorXf I = VectorXf::LinSpaced(t_grid.size(), 0, t_grid.size()-1);
            std::sort(I.data(), I.data() + I.size(), [&t_grid] (size_t i1, size_t i2) {return t_grid[i1] < t_grid[i2];} );
            VectorXf xx(I.size()), yy(I.size());
            for(int k=0; k<t_grid.size(); k++){
                xx(k) = xx_temp(I(k));
                yy(k) = yy_temp(I(k));
            }
            
            // Get rid of points that are outside the image grid.
            I.resize(xx.size()), I.setZero();
            float vol_boundary = Nside/2.0 * pixelWidth;
            for(int k=0; k<xx.size(); k++)
            {
                if(xx(k) >= -vol_boundary && xx(k) <= vol_boundary)
                {    if(yy(k) >= -vol_boundary && yy(k) <= vol_boundary)
                     {
                        I(k) = 1.0;
                     }
                    
                }
            }
            removeBadElements(xx, yy, I);
            
            //If the ray pass through the image grid
            if (xx.size() != 0 && yy.size() != 0)
                
                //Get rid of double counted points.
                I.resize(xx.size()), I.setOnes();
                for(int k=0; k<xx.size()-1; k++)
                {
                    if(abs(xx(k+1) - xx(k)) <= 1e-4)
                    {    if(abs(yy(k+1) - yy(k)) <= 1e-4)
                         {
                             I(k) = 0;
                         }
                    }
                }
                removeBadElements(xx, yy, I);
            
                //Calculate the length within the cell.
                tne = xx.size() - 1;
                VectorXf length(tne);
                for(int k=0; k<tne; k++)
                {
                    length(k) = sqrt( pow((xx(k+1) - xx(k)),2) + pow((yy(k+1) - yy(k)),2) );
                }
                int numvals = length.size();

                //Remove the rays that are on the boundary of the box in the
                //top or to the right of the image grid
                bool check1, check2, check;
                check1 = check2 = false;
                check = true;
                if (b == 0 && abs(yrayRoated(j) - Nside/2 * pixelWidth) < 1e-15) {
                    check1 = true;
                }
                if (a == 0 && abs(xrayRoated(j) - Nside/2 * pixelWidth) < 1e-15) {
                    check2 = true;
                }
                if (check1 || check2) {
                    check = false;
                }
            
                //Calculate corresponding indices in measurement matrix
                if(numvals > 0 && check == true)
                {
                    
                    tne = xx.size() - 1;
                    VectorXf midpointsX(tne);
                    VectorXf midpointsY(tne);
                    for(int k=0; k<tne; k++)
                    {
                        midpointsX(k) = 0.5 * (xx(k) + xx(k+1));
                        midpointsY(k) = 0.5 * (yy(k) + yy(k+1));
                    }
            
                    midpointsX = (midpointsX.array().abs() < 1e-10).select(0, midpointsX);
                    midpointsY = (midpointsY.array().abs() < 1e-10).select(0, midpointsY);
            
                    //Calculate the pixel index for mid points
                    VectorXf pixelIndex(tne);
                    pixelIndex = floor((Nside/2 - midpointsY.array()/pixelWidth))*Nside + floor((midpointsX.array()/pixelWidth + Nside/2));
                    pixelIndex = (pixelIndex.array() < 0 ).select(0,pixelIndex);
                    
                    //Create the indices to store the values to the measurement matrix
                    int idxstart = idxend;
                    idxend = idxstart + numvals;
                    int idx = 0;

                    //Store row numbers, column numbers and values
                    for(int k=idxstart; k<idxend; k++)
                    {
                        rows(k) = i * Nray + j;
                        cols(k) = pixelIndex(idx);
                        vals(k) = length(idx);
                        idx = idx + 1;
                    }
                }
        }
        
    }
    
    //Truncate excess zeros.
    for(int i=0; i <idxend; i++)
    {
        A.insert(rows(i), cols(i)) = vals(i);
    }
    A.makeCompressed();
    
}

float rmepsilon(float input)
{
    if (abs(input) < 1e-10)
        input = 0;
    return input;
}

void removeBadElements(Eigen::VectorXf& xx, Eigen::VectorXf& yy, Eigen::VectorXf I)
{
    //Remove elements at indices where I == 0.
    VectorXf xx_temp(xx.size()), yy_temp(xx.size());
    int ind = 0;
    for(int k=0; k<xx.size();k++)
    {
        if(I(k) != 0)
        {
            xx_temp(ind) = xx(k);
            yy_temp(ind) = yy(k);
            ind++;
        }
    }
    xx.resize(ind), yy.resize(ind);
    xx = xx_temp.head(ind);
    yy = yy_temp.head(ind);
}

void saveResults(Eigen::VectorXf vec, int direc, std::string name)
{
    std::ofstream outfile( "Results/" + to_string(direc) + "/" + name + ".txt");
    for (int i=0; i < vec.size(); i++)
    {
        outfile << vec(i) << "\n";
    }
}

void saveVec(Eigen::VectorXf vec, std::string name)
{
    std::ofstream outfile( "Results/ASD_tv/" + name + ".txt");
    for (int i=0; i < vec.size(); i++)
    {
        outfile << vec(i) << "\n";
    }
}

void read_parameters(int& Niter,float& Niter_red,int& ng,float& dTheta,float& beta,float& beta_red,float& alpha,float& alpha_red,float& eps,float& r_max)
{
    //Read values in Parameters.txt file. 
    ifstream parametersFile;
    parametersFile.open("parameters.txt");
    string input[30];
    int i = 0;
    string text;
    while(!parametersFile.eof())
    {
        getline(parametersFile,text);
        input[i++]=text;
    }
    parametersFile.close();

    // Convert Strings into Int/Float. 
    Niter = stoi(input[1]);
    Niter_red = stof(input[4]);
    ng = stoi(input[7]);
    dTheta = stof(input[10]);
    beta = stof(input[13]);
    beta_red = stof(input[16]);
    alpha = stof(input[19]);
    alpha_red = stof(input[22]);
    eps = stof(input[25]);
    r_max = stof(input[28]);
    

}

void poissonNoise(Eigen::VectorXf& b, int Nc)
{
    VectorXf temp_b(b.size());
    temp_b = b;
    float mean = b.mean();
    float N = b.sum();
    b  = b / ( b.sum() ) * Nc * b.size();
    std::default_random_engine generator;
    for(int i=0; i < b.size(); i++)
    {
        std::poisson_distribution<int> distribution(b(i));
        b(i) = distribution(generator);
        
    }
    b = b / ( Nc * b.size() ) * N;
    temp_b = temp_b.array() - b.array();
    float std = sqrt( ( temp_b.array() - temp_b.mean() ).square().sum() / (temp_b.size() - 1) );
    float SNR = mean/std;
    cout << SNR << endl;
}
