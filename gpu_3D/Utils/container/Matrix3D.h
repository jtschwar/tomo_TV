#ifndef matrix_3d_hpp
#define matrix_3d_hpp

#include <Eigen/Core>

class Matrix3D
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
    
public:
    
    int nx, ny, nz, size;
    int gpuIndex = -1;
    
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

    // Calculate L2 and L1 Norms for Volume
    float norm();   float l1_norm();

    // Sum All the Values in the Reconstruction
    float sum();

    // Apply Positivity (Remove Negative Voxels)
    void positivity();

    // Set Background Equal to Value
    void setBackground(int backgroundValue=1);

    // Apply Softhreshold Operation to Volume
    void soft_threshold(float lambda);
};

#endif
