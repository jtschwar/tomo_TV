/*shared macros*/
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return (float)(x*x);
        }
};

template <typename T>
struct absolute_value 
{
  __host__ __device__ 
    T operator()(const T &x) const {
        return x < T(0) ? -x : x;
    }
};


/*checks CUDA call, should be used in functions returning <int> value
if error happens, writes to standard error and explicitly returns -1*/
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        return -1;                                                             \
    }                                                                          \
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        return -1;                                                                \
    }                                                                          \
}
