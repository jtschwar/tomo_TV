#ifndef matrix_3d_hpp
#define matrix_3d_hpp

class Matrix3D
{

public:
    
    int nx, ny, nz, size;
    
    float *data;
    
    // Constructors
    Matrix3D();
    Matrix3D(int Nx, int Ny, int Nz);

    // Access Data
    float get_val(int i,int j,int k);
    
    // Calculate Index
    int calc_index(int i, int j, int k);

    float norm();

    float sum();

};

#endif
