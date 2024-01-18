#ifndef matrix_4d_hpp
#define matrix_4d_hpp

#include <Eigen/Core>

class Matrix4D
{

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;    

public:
    
    int nx, ny, nz, nel, volSize;
    int gpuIndex = -1;
    
    float *data;
    
    // Constructors
    Matrix4D();
    Matrix4D(int nel, int Nx, int Ny, int Nz);
    
    // Setter Functions
    void setData1D(Vec inBuffer, int element, int slice);
    void setData2D(Mat inBuffer, int element, int slice);

    // Getter Functions
    float *getData1D(int element, int slice);
    Mat getData2D(int element, int slice);

    // Access Data
    float get_val(int e, int i,int j,int k);
    
    // Calculate Index
    int index(int e, int i, int j, int k);

    void positivity();
};

#endif
