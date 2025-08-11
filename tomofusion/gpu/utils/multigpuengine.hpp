//
//  multigpuengine.hpp
//
//  Created by Hovden Group on 8/10/25.
//  Multi-GPU Tomography Engine
//

#ifndef MULTIGPUENGINE_HPP
#define MULTIGPUENGINE_HPP

#include "tomoengine.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <omp.h>

// Forward declaration
struct GPUContext;

class multigpuengine : public tomoengine {
private:
    
    // Multi-GPU specific
    std::vector<int> gpu_ids;
    std::vector<std::unique_ptr<GPUContext>> gpu_contexts;
    mutable std::mutex context_mutex;
    static thread_local GPUContext* tls_context;
    
    // Helper methods
    GPUContext* get_thread_context() const;

public:
    // Constructors (inherit from tomoengine)
    multigpuengine(int Ns, int Nray);
    multigpuengine(int Ns, int Nray, Eigen::VectorXf pyAngles);
    
    // Destructor
    ~multigpuengine();
    
    // These methods exist in tomoengine but might not be virtual, so don't use override
    void SART(float beta, int nIter);
    void SIRT(int nIter);
    void CGLS(int nIter);
    void FBP(bool apply_positivity);
    float poisson_ML(float lambda);
    void forwardProjection();
    
    // Multi-GPU specific methods
    std::vector<int> get_gpu_ids() const;
    bool is_multi_gpu_enabled() const;
    void print_gpu_usage() const;
};

#endif // MULTIGPUENGINE_HPP