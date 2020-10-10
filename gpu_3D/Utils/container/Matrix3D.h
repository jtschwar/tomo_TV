#ifndef matrix_3d_hpp
#define matrix_3d_hpp

#include <Eigen/Core>

class Matrix3D
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
    
public:
    
    int nx, ny, nz, size, gpuIndex;
    
    float *data;
    
    // Constructors
    Matrix3D();
    Matrix3D(int Nx, int Ny, int Nz);
    
    void setData(Mat inBuffer, int slice);
    Mat getData(int slice);

    // Access Data
    float get_val(int i,int j,int k);
    
    // Calculate Index
    int index(int i, int j, int k);

    float norm();

    float sum();

    void positivity();
};

#endif
