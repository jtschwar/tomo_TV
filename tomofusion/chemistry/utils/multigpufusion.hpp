//
//  multigpufusion.hpp
//
//  Created by Hovden Group on 8/10/25.
//  Multi-GPU Multimodal Fusion Engine
//

#ifndef MULTIGPUFUSION_HPP
#define MULTIGPUFUSION_HPP

#include "multimodal.hpp"
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <omp.h>

// Forward declaration
struct FusionGPUContext;

class multigpufusion : public multimodal {
private:
    
    // Multi-GPU specific
    std::vector<int> gpu_ids;
    std::vector<std::unique_ptr<FusionGPUContext>> gpu_contexts;
    mutable std::mutex context_mutex;
    static thread_local FusionGPUContext* tls_context;
    
    // Helper methods
    FusionGPUContext* get_thread_context() const;

public:
    // Constructors (inherit from multimodal)
    multigpufusion(int Ns, int Nray, int Nelements);
    multigpufusion(int Ns, int Nray, int Nelements, Eigen::VectorXf haadfAngles, Eigen::VectorXf chemAngles);
    
    // Destructor
    ~multigpufusion();
    
    // Override key multimodal methods for multi-GPU
    float poisson_ml(float lambdaCHEM) override;
    std::tuple<float,float> data_fusion(float lambdaHAADF, float lambdaCHEM, int nIter, std::string method) override;
    void chemical_SART(int nIter) override;
    void chemical_SIRT(int nIter) override;
    float data_distance() override;
    
    // Multi-GPU specific methods
    std::vector<int> get_gpu_ids() const;
    bool is_multi_gpu_enabled() const;
    void print_gpu_usage() const;
};

#endif // MULTIGPUFUSION_HPP