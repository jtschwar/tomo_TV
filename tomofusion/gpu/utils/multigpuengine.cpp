//
//  multigpuengine.cpp
//
//  Created by Hovden Group on 8/10/25.
//  Multi-GPU Tomography Engine Implementation
//

#include "multigpuengine.hpp"
#include "regularizers/tv_gd.h"
#include "regularizers/tv_fgp.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <set>

//Python bindings for multigpuengine module.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

// Thread-local GPU context
struct GPUContext {
    int gpu_id;
    bool initialized = false;
    
    // ASTRA algorithm objects
    std::unique_ptr<CCudaSartAlgorithm> sart_algo;
    std::unique_ptr<CCudaSirtAlgorithm> sirt_algo;
    std::unique_ptr<CCudaCglsAlgorithm> cgls_algo;
    std::unique_ptr<CCudaFilteredBackProjectionAlgorithm> fbp_algo;
    std::unique_ptr<CCudaForwardProjectionAlgorithm> fp_algo;
    std::unique_ptr<CCudaBackProjectionAlgorithm> bp_algo;
    
    // ASTRA data objects
    std::unique_ptr<CFloat32VolumeData2D> vol;
    std::unique_ptr<CFloat32ProjectionData2D> sino;
    
    GPUContext(int id) : gpu_id(id) {}
    
    void initialize(const CVolumeGeometry2D* vol_geom, 
                   const CParallelProjectionGeometry2D* proj_geom,
                   CCudaProjector2D* proj) {
        if (initialized) return;
        
        cudaSetDevice(gpu_id);
        
        // Create GPU-specific ASTRA objects
        vol = std::make_unique<CFloat32VolumeData2D>(*vol_geom);
        sino = std::make_unique<CFloat32ProjectionData2D>(*proj_geom);
        
        // Initialize algorithms
        sart_algo = std::make_unique<CCudaSartAlgorithm>();
        sart_algo->initialize(proj, sino.get(), vol.get());
        sart_algo->setConstraints(true, 0, false, 1);
        sart_algo->setGPUIndex(gpu_id);
        
        sirt_algo = std::make_unique<CCudaSirtAlgorithm>();
        sirt_algo->initialize(proj, sino.get(), vol.get());
        sirt_algo->setConstraints(true, 0, false, 1);
        sirt_algo->setGPUIndex(gpu_id);
        
        cgls_algo = std::make_unique<CCudaCglsAlgorithm>();
        cgls_algo->initialize(proj, sino.get(), vol.get());
        cgls_algo->setGPUIndex(gpu_id);
        
        fbp_algo = std::make_unique<CCudaFilteredBackProjectionAlgorithm>();
        fbp_algo->setGPUIndex(gpu_id);
        
        fp_algo = std::make_unique<CCudaForwardProjectionAlgorithm>();
        fp_algo->initialize(proj, vol.get(), sino.get());
        fp_algo->setGPUIndex(gpu_id);
        
        bp_algo = std::make_unique<CCudaBackProjectionAlgorithm>();
        bp_algo->initialize(proj, sino.get(), vol.get());
        bp_algo->setGPUIndex(gpu_id);
        
        initialized = true;
    }
};

// Thread-local storage definition
thread_local GPUContext* multigpuengine::tls_context = nullptr;

// Initialize Empty Volume Constructor
multigpuengine::multigpuengine(int Ns, int Nray) : tomoengine(Ns, Nray) {
    // Auto-detect and use ALL available GPUs
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    
    if (gpu_count > 0) {
        for (int i = 0; i < gpu_count; i++) {
            gpu_ids.push_back(i);
        }
        std::cout << "MultiGPU Engine initialized (empty) with ALL " << gpu_count << " GPUs: ";
    } else {
        gpu_ids.push_back(0);  // Fallback to GPU 0
        std::cout << "Warning: No GPUs detected, using GPU 0: ";
    }
    
    gpu_contexts.reserve(gpu_ids.size() * 4);
    
    for (int gpu : gpu_ids) std::cout << gpu << " ";
    std::cout << std::endl;
}

// Tomography Constructor
multigpuengine::multigpuengine(int Ns, int Nray, Eigen::VectorXf pyAngles) : tomoengine(Ns, Nray, pyAngles) {
    // Auto-detect and use ALL available GPUs
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    
    if (gpu_count > 0) {
        for (int i = 0; i < gpu_count; i++) {
            gpu_ids.push_back(i);
        }
        std::cout << "MultiGPU Engine initialized with ALL " << gpu_count << " GPUs: ";
    } else {
        gpu_ids.push_back(0);  // Fallback to GPU 0
        std::cout << "Warning: No GPUs detected, using GPU 0: ";
    }
    
    gpu_contexts.reserve(gpu_ids.size() * 4);
    
    for (int gpu : gpu_ids) std::cout << gpu << " ";
    std::cout << std::endl;
}

// Destructor
multigpuengine::~multigpuengine() {
    // Base class destructor handles ASTRA cleanup
}

// Get thread-local GPU context
GPUContext* multigpuengine::get_thread_context() const {
    if (!tls_context) {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        // Use OpenMP thread number for more predictable assignment
        int omp_thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[omp_thread_id % gpu_ids.size()];
        
        // Create new context
        auto context = std::make_unique<GPUContext>(assigned_gpu);
        context->initialize(vol_geom, proj_geom, proj);
        
        tls_context = context.get();
        const_cast<multigpuengine*>(this)->gpu_contexts.push_back(std::move(context));
    }
    return tls_context;
}

// Override Multi-GPU Reconstruction Algorithms
void multigpuengine::SART(float beta, int nIter) {
    
    int Nproj = Nrow / Ny;
    
    #pragma omp parallel num_threads(gpu_ids.size())
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        // Get thread-local GPU context
        GPUContext* ctx = get_thread_context();
        
        // Make sure ASTRA algorithms use the correct GPU
        ctx->sart_algo->setGPUIndex(assigned_gpu);
        
        // Configure SART parameters  
        ctx->sart_algo->updateProjOrder(projOrder);
        if (beta != 1) {
            ctx->sart_algo->setRelaxationParameter(beta);
        }
        
        // Distribute slices across threads
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            
            ctx->sino->copyData((float32*) &b(s,0));
            ctx->vol->copyData((float32*) &recon.data[recon.index(s,0,0)]);
            
            ctx->sart_algo->updateSlice(ctx->sino.get(), ctx->vol.get());
            ctx->sart_algo->run(Nproj * nIter);
            
            memcpy(&recon.data[recon.index(s,0,0)], ctx->vol->getData(), 
                   sizeof(float)*Ny*Nz);
        }
    }
}

void multigpuengine::SIRT(int nIter) {
    #pragma omp parallel num_threads(gpu_ids.size())
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        GPUContext* ctx = get_thread_context();
        
        // Set GPU for SIRT algorithm
        ctx->sirt_algo->setGPUIndex(assigned_gpu);
        
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            ctx->sino->copyData((float32*) &b(s,0));
            if (momentum) {
                ctx->vol->copyData((float32*) &yk.data[yk.index(s,0,0)]);
            } else {
                ctx->vol->copyData((float32*) &recon.data[recon.index(s,0,0)]);
            }
            
            ctx->sirt_algo->updateSlice(ctx->sino.get(), ctx->vol.get());
            ctx->sirt_algo->run(nIter);
            
            if (momentum) {
                memcpy(&yk.data[yk.index(s,0,0)], ctx->vol->getData(), 
                       sizeof(float)*Ny*Nz);
            } else {
                memcpy(&recon.data[recon.index(s,0,0)], ctx->vol->getData(), 
                       sizeof(float)*Ny*Nz);
            }
        }
    }
}

void multigpuengine::CGLS(int nIter) {
    #pragma omp parallel num_threads(gpu_ids.size())
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        GPUContext* ctx = get_thread_context();
        
        // Set GPU for CGLS algorithm
        ctx->cgls_algo->setGPUIndex(assigned_gpu);
        
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            ctx->sino->copyData((float32*) &b(s,0));
            ctx->vol->copyData((float32*) &recon.data[recon.index(s,0,0)]);
            
            ctx->cgls_algo->initialize(proj, ctx->sino.get(), ctx->vol.get());
            ctx->cgls_algo->run(nIter);
            
            memcpy(&recon.data[recon.index(s,0,0)], ctx->vol->getData(), 
                   sizeof(float)*Ny*Nz);
        }
    }
    
    recon.positivity();
}

void multigpuengine::FBP(bool apply_positivity) {
    E_FBPFILTER fbfFilt = convertStringToFilter(fbfFilter);
    
    #pragma omp parallel num_threads(gpu_ids.size())
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        GPUContext* ctx = get_thread_context();
        
        // Set GPU for FBP algorithm
        ctx->fbp_algo->setGPUIndex(assigned_gpu);
        
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            ctx->sino->copyData((float32*) &b(s,0));
            ctx->vol->copyData((float32*) &recon.data[recon.index(s,0,0)]);
            
            ctx->fbp_algo->initialize(ctx->sino.get(), ctx->vol.get(), fbfFilt);
            ctx->fbp_algo->run();
            
            memcpy(&recon.data[recon.index(s,0,0)], ctx->vol->getData(), 
                   sizeof(float)*Ny*Nz);
        }
    }
    
    if (apply_positivity) { 
        recon.positivity(); 
    }
}

void multigpuengine::forwardProjection() {
    #pragma omp parallel num_threads(gpu_ids.size())
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        GPUContext* ctx = get_thread_context();
        
        // Set GPU for forward projection algorithm
        ctx->fp_algo->setGPUIndex(assigned_gpu);
        
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            ctx->vol->copyData((float32*) &recon.data[recon.index(s,0,0)]);
            
            ctx->fp_algo->updateSlice(ctx->sino.get(), ctx->vol.get());
            ctx->fp_algo->run();
            
            memcpy(&g(s,0), ctx->sino->getData(), sizeof(float)*Nrow);
        }
    }
}

float multigpuengine::poisson_ML(float lambda) {
    float total_cost = 0;
    float eps = 1e-1;
    
    #pragma omp parallel num_threads(gpu_ids.size()) reduction(+:total_cost)
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        GPUContext* ctx = get_thread_context();
        
        // Set GPU for forward/back projection algorithms
        ctx->fp_algo->setGPUIndex(assigned_gpu);
        ctx->bp_algo->setGPUIndex(assigned_gpu);
        
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            Eigen::VectorXf xx_local(Ny*Nz), Ax_local(Nrow), updateML_local(Ny*Nz);
            
            memcpy(&xx_local(0), &recon.data[recon.index(s,0,0)], sizeof(float)*Ny*Nz);
            
            // Forward projection
            ctx->vol->copyData((float32*) &xx_local(0));
            ctx->fp_algo->initialize(proj, ctx->vol.get(), ctx->sino.get());
            ctx->fp_algo->run();
            memcpy(&Ax_local(0), ctx->sino->getData(), sizeof(float)*Nrow);
            
            // Back projection for gradient
            Eigen::VectorXf residual = (Ax_local - b.row(s).transpose()).array() / (Ax_local.array() + eps).array();
            ctx->sino->copyData((float32*) &residual(0));
            ctx->bp_algo->initialize(proj, ctx->sino.get(), ctx->vol.get());
            ctx->bp_algo->run();
            memcpy(&updateML_local(0), ctx->vol->getData(), sizeof(float)*Ny*Nz);
            
            // Update
            xx_local -= (lambda / L_Aml) * updateML_local;
            
            // Copy back to reconstruction
            memcpy(&recon.data[recon.index(s,0,0)], &xx_local(0), sizeof(float)*Ny*Nz);
            
            // Calculate cost for this slice
            float slice_cost = (Ax_local.array() - b.row(s).transpose().array() * 
                               (Ax_local.array() + eps).log().array()).sum();
            
            total_cost += slice_cost;
        }
    }
    
    recon.positivity();
    return total_cost;
}

// Multi-GPU specific utility methods
std::vector<int> multigpuengine::get_gpu_ids() const {
    return gpu_ids;
}

bool multigpuengine::is_multi_gpu_enabled() const {
    return gpu_ids.size() > 1;
}

void multigpuengine::print_gpu_usage() const {
    std::cout << "Active GPU contexts: " << gpu_contexts.size() << std::endl;
    std::cout << "GPU IDs in use: ";
    std::set<int> unique_gpus;
    for (const auto& ctx : gpu_contexts) {
        unique_gpus.insert(ctx->gpu_id);
    }
    for (int gpu_id : unique_gpus) {
        std::cout << gpu_id << " ";
    }
    std::cout << std::endl;
}

PYBIND11_MODULE(multigpuengine, m)
{
    m.doc() = "Multi-GPU Tomography Engine using OpenMP";
    
    py::class_<multigpuengine, tomoengine>(m, "multigpuengine")
        .def(py::init<int, int>(),
             "Empty volume constructor",
             py::arg("Ns"), py::arg("Nray"))
        .def(py::init<int, int, Eigen::VectorXf>(),
             "Tomography constructor with angles",
             py::arg("Ns"), py::arg("Nray"), py::arg("pyAngles"))
        // ADD THESE OVERRIDDEN METHODS:
        .def("SART", &multigpuengine::SART, 
             "Multi-GPU SART Reconstruction",
             py::arg("beta"), py::arg("nIter"))
        .def("SIRT", &multigpuengine::SIRT,
             "Multi-GPU SIRT Reconstruction", 
             py::arg("nIter"))
        .def("CGLS", &multigpuengine::CGLS,
             "Multi-GPU CGLS Reconstruction",
             py::arg("nIter"))
        .def("FBP", &multigpuengine::FBP,
             "Multi-GPU FBP Reconstruaaction",
             py::arg("apply_positivity"))
        .def("forward_projection", &multigpuengine::forwardProjection,
             "Multi-GPU Forward Projection")
        .def("poisson_ML", &multigpuengine::poisson_ML,
             "Multi-GPU Poisson ML Reconstruction",
             py::arg("lambda"))
        // Utility methods:
        .def("get_gpu_ids", &multigpuengine::get_gpu_ids,
             "Get list of GPU IDs being used")
        .def("is_multi_gpu_enabled", &multigpuengine::is_multi_gpu_enabled,
             "Check if multi-GPU parallelization is enabled")
        .def("print_gpu_usage", &multigpuengine::print_gpu_usage,
             "Print current GPU usage information");
}