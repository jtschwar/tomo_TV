//
//  multigpufusion.cpp
//
//  Created by Hovden Group on 8/10/25.
//  Multi-GPU Multimodal Fusion Engine Implementation
//

#include "multigpufusion.hpp"
#include "regularizers/tv_gd.h"
#include "regularizers/tv_fgp.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <set>

//Python bindings for multigpufusion module.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Mat;
typedef Eigen::VectorXf Vec;

// Thread-local GPU context for multimodal fusion
struct FusionGPUContext {
    int gpu_id;
    bool initialized = false;
    
    // ASTRA algorithm objects for both HAADF and Chemical projections
    std::unique_ptr<CCudaSartAlgorithm> sart_algo;
    std::unique_ptr<CCudaSirtAlgorithm> sirt_algo;
    std::unique_ptr<CCudaFilteredBackProjectionAlgorithm> fbp_algo;
    std::unique_ptr<CCudaForwardProjectionAlgorithm> fp_algo;
    std::unique_ptr<CCudaBackProjectionAlgorithm> bp_algo;
    
    // ASTRA data objects for both geometries
    std::unique_ptr<CFloat32VolumeData2D> vol;
    std::unique_ptr<CFloat32ProjectionData2D> haadf_sino;
    std::unique_ptr<CFloat32ProjectionData2D> chem_sino;
    
    FusionGPUContext(int id) : gpu_id(id) {}
    
    void initialize(const CVolumeGeometry2D* vol_geom, 
                   const CParallelProjectionGeometry2D* haadf_proj_geom,
                   const CParallelProjectionGeometry2D* chem_proj_geom,
                   CCudaProjector2D* h_proj,
                   CCudaProjector2D* c_proj) {
        if (initialized) return;
        
        cudaSetDevice(gpu_id);
        
        // Create GPU-specific ASTRA objects
        vol = std::make_unique<CFloat32VolumeData2D>(*vol_geom);
        haadf_sino = std::make_unique<CFloat32ProjectionData2D>(*haadf_proj_geom);
        chem_sino = std::make_unique<CFloat32ProjectionData2D>(*chem_proj_geom);
        
        // Initialize algorithms for HAADF projections
        sart_algo = std::make_unique<CCudaSartAlgorithm>();
        sart_algo->initialize(h_proj, haadf_sino.get(), vol.get());
        sart_algo->setConstraints(true, 0, false, 1);
        sart_algo->setGPUIndex(gpu_id);
        
        sirt_algo = std::make_unique<CCudaSirtAlgorithm>();
        sirt_algo->initialize(h_proj, haadf_sino.get(), vol.get());
        sirt_algo->setConstraints(true, 0, false, 1);
        sirt_algo->setGPUIndex(gpu_id);
        
        fbp_algo = std::make_unique<CCudaFilteredBackProjectionAlgorithm>();
        fbp_algo->setGPUIndex(gpu_id);
        
        fp_algo = std::make_unique<CCudaForwardProjectionAlgorithm>();
        fp_algo->initialize(h_proj, vol.get(), haadf_sino.get());
        fp_algo->setGPUIndex(gpu_id);
        
        bp_algo = std::make_unique<CCudaBackProjectionAlgorithm>();
        bp_algo->initialize(h_proj, haadf_sino.get(), vol.get());
        bp_algo->setGPUIndex(gpu_id);
        
        initialized = true;
    }
};

// Thread-local storage definition
thread_local FusionGPUContext* multigpufusion::tls_context = nullptr;

// Initialize Empty Volume Constructor
multigpufusion::multigpufusion(int Ns, int Nray, int Nelements) : multimodal(Ns, Nray, Nelements) {
    // Auto-detect and use ALL available GPUs
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    
    if (gpu_count > 0) {
        for (int i = 0; i < gpu_count; i++) {
            gpu_ids.push_back(i);
        }
        std::cout << "MultiGPU Fusion Engine initialized (empty) with ALL " << gpu_count << " GPUs: ";
    } else {
        gpu_ids.push_back(0);  // Fallback to GPU 0
        std::cout << "Warning: No GPUs detected, using GPU 0: ";
    }
    
    gpu_contexts.reserve(gpu_ids.size() * 4);
    
    for (int gpu : gpu_ids) std::cout << gpu << " ";
    std::cout << std::endl;
}

// Multimodal Constructor
multigpufusion::multigpufusion(int Ns, int Nray, int Nelements, Eigen::VectorXf haadfAngles, Eigen::VectorXf chemAngles) 
    : multimodal(Ns, Nray, Nelements, haadfAngles, chemAngles) {
    // Auto-detect and use ALL available GPUs
    int gpu_count;
    cudaGetDeviceCount(&gpu_count);
    
    if (gpu_count > 0) {
        for (int i = 0; i < gpu_count; i++) {
            gpu_ids.push_back(i);
        }
        std::cout << "MultiGPU Fusion Engine initialized with ALL " << gpu_count << " GPUs: ";
    } else {
        gpu_ids.push_back(0);  // Fallback to GPU 0
        std::cout << "Warning: No GPUs detected, using GPU 0: ";
    }
    
    gpu_contexts.reserve(gpu_ids.size() * 4);
    
    for (int gpu : gpu_ids) std::cout << gpu << " ";
    std::cout << std::endl;
}

// Destructor
multigpufusion::~multigpufusion() {
    // Base class destructor handles ASTRA cleanup
}

// Get thread-local GPU context
FusionGPUContext* multigpufusion::get_thread_context() const {
    if (!tls_context) {
        std::lock_guard<std::mutex> lock(context_mutex);
        
        // Use OpenMP thread number for more predictable assignment
        int omp_thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[omp_thread_id % gpu_ids.size()];
        
        // Create new context
        auto context = std::make_unique<FusionGPUContext>(assigned_gpu);
        context->initialize(vol_geom, haadfProjGeom, chemProjGeom, hProj, cProj);
        
        tls_context = context.get();
        const_cast<multigpufusion*>(this)->gpu_contexts.push_back(std::move(context));
    }
    return tls_context;
}

// Multi-GPU Poisson ML Reconstruction
float multigpufusion::poisson_ml(float lambdaCHEM) {
    float total_cost = 0;
    
    #pragma omp parallel num_threads(gpu_ids.size()) reduction(+:total_cost)
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        // Get thread-local GPU context
        FusionGPUContext* ctx = get_thread_context();
        
        // Set GPU for algorithms
        ctx->fp_algo->setGPUIndex(assigned_gpu);
        ctx->bp_algo->setGPUIndex(assigned_gpu);
        
        // Distribute slices across threads
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            Eigen::VectorXf xx_local(Ny*Nz*Nel), Ax_local(NrowChem*Nel), updateCHEM_local(Ny*Nz*Nel);
            
            // Concatenate elements for this slice
            for (int e = 0; e < Nel; e++) {
                memcpy(&xx_local(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz);
            }
            
            // Forward projection for all elements (use chemical geometry)
            for (int e = 0; e < Nel; e++) {
                ctx->vol->copyData((float32*) &xx_local(e*Ny*Nz));
                ctx->fp_algo->initialize(cProj, ctx->vol.get(), ctx->chem_sino.get());
                ctx->fp_algo->run();
                memcpy(&Ax_local(e*NrowChem), ctx->chem_sino->getData(), sizeof(float)*NrowChem);
            }
            
            // Back projection for gradient (each element separately)
            for (int e = 0; e < Nel; e++) {
                Eigen::VectorXf residual = (Ax_local.segment(e*NrowChem, NrowChem) - 
                                          bChem.row(s).segment(e*NrowChem, NrowChem).transpose()).array() / 
                                          (Ax_local.segment(e*NrowChem, NrowChem).array() + eps).array();
                
                ctx->chem_sino->copyData((float32*) &residual(0));
                ctx->bp_algo->initialize(cProj, ctx->chem_sino.get(), ctx->vol.get());
                ctx->bp_algo->run();
                memcpy(&updateCHEM_local(e*Ny*Nz), ctx->vol->getData(), sizeof(float)*Ny*Nz);
            }
            
            // Update
            xx_local -= (lambdaCHEM / L_Aps) * updateCHEM_local;
            
            // Copy back to reconstruction
            for (int e = 0; e < Nel; e++) {
                memcpy(&recon.data[recon.index(e,s,0,0)], &xx_local(e*Ny*Nz), sizeof(float)*Ny*Nz);
            }
            
            // Calculate cost for this slice
            if (measureChem) {
                float slice_cost = (Ax_local.array() - bChem.row(s).transpose().array() * 
                                   (Ax_local.array() + eps).log().array()).sum();
                total_cost += slice_cost;
            }
        }
    }
    
    recon.positivity();
    return total_cost;
}

// Multi-GPU Data Fusion
std::tuple<float,float> multigpufusion::data_fusion(float lambdaHAADF, float lambdaCHEM, int nIter, std::string method) {
    float total_costHAADF = 0;
    float total_costCHEM = 0;
    
    // Initialize algorithms if needed
    if (method == "SART" && (algo_sart == NULL)) { initializeSART("sequential"); }
    else if (method == "SIRT" && (algo_sirt == NULL)) { initializeSIRT(); }
    
    #pragma omp parallel num_threads(gpu_ids.size()) reduction(+:total_costCHEM)
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        // Get thread-local GPU context
        FusionGPUContext* ctx = get_thread_context();
        
        // Set GPU for algorithms
        ctx->fp_algo->setGPUIndex(assigned_gpu);
        ctx->bp_algo->setGPUIndex(assigned_gpu);
        ctx->sart_algo->setGPUIndex(assigned_gpu);
        ctx->sirt_algo->setGPUIndex(assigned_gpu);
        
        // Distribute slices across threads
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            Eigen::VectorXf xx_local(Ny*Nz*Nel), Ax_local(NrowChem*Nel);
            Eigen::VectorXf updateCHEM_local(Ny*Nz*Nel), updateHAADF_local(Ny*Nz*Nel);
            Eigen::VectorXf modelHAADF_local(Nslice*Ny), updateVol_local(Nslice*Ny);
            
            // Concatenate elements for this slice
            for (int e = 0; e < Nel; e++) {
                memcpy(&xx_local(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz);
            }
            
            // Compute HAADF gradient update
            if (gamma == 1) {
                modelHAADF_local = sigma * xx_local;
            } else {
                modelHAADF_local = sigma * (Eigen::VectorXf(xx_local.array().pow(gamma)));
            }
            
            // Forward project HAADF model
            ctx->vol->copyData((float32*) &modelHAADF_local(0));
            ctx->fp_algo->initialize(hProj, ctx->vol.get(), ctx->haadf_sino.get());
            ctx->fp_algo->run();
            memcpy(&g(s,0), ctx->haadf_sino->getData(), sizeof(float)*NrowHaadf);
            
            // HAADF reconstruction step (fuse operation)
            ctx->haadf_sino->copyData((float32*) &bh(s,0));
            ctx->vol->copyData((float32*) &modelHAADF_local(0));
            
            if (method == "SART") {
                int Nproj = NrowHaadf / Ny;
                ctx->sart_algo->updateSlice(ctx->haadf_sino.get(), ctx->vol.get());
                ctx->sart_algo->run(Nproj * nIter);
            } else {
                ctx->sirt_algo->updateSlice(ctx->haadf_sino.get(), ctx->vol.get());
                ctx->sirt_algo->run(nIter);
            }
            
            memcpy(&updateVol_local(0), ctx->vol->getData(), sizeof(float)*Nslice*Ny);
            
            // Back propagate to individual chemistries
            if (gamma == 1) {
                updateHAADF_local = sigma.transpose() * (updateVol_local - modelHAADF_local);
            } else {
                // Simplified gamma handling
                updateHAADF_local = gamma * sigma.transpose() * (updateVol_local - modelHAADF_local);
            }
            
            // Chemical ML update (Poisson-ML) - same as poisson_ml
            for (int e = 0; e < Nel; e++) {
                ctx->vol->copyData((float32*) &xx_local(e*Ny*Nz));
                ctx->fp_algo->initialize(cProj, ctx->vol.get(), ctx->chem_sino.get());
                ctx->fp_algo->run();
                memcpy(&Ax_local(e*NrowChem), ctx->chem_sino->getData(), sizeof(float)*NrowChem);
                
                Eigen::VectorXf residual = (Ax_local.segment(e*NrowChem, NrowChem) - 
                                          bChem.row(s).segment(e*NrowChem, NrowChem).transpose()).array() / 
                                          (Ax_local.segment(e*NrowChem, NrowChem).array() + eps).array();
                
                ctx->chem_sino->copyData((float32*) &residual(0));
                ctx->bp_algo->initialize(cProj, ctx->chem_sino.get(), ctx->vol.get());
                ctx->bp_algo->run();
                memcpy(&updateCHEM_local(e*Ny*Nz), ctx->vol->getData(), sizeof(float)*Ny*Nz);
            }
            
            // Combined update
            xx_local -= lambdaCHEM/L_Aps * updateCHEM_local - lambdaHAADF * updateHAADF_local;
            
            // Copy back to reconstruction
            for (int e = 0; e < Nel; e++) {
                memcpy(&recon.data[recon.index(e,s,0,0)], &xx_local(e*Ny*Nz), sizeof(float)*Ny*Nz);
            }
            
            // Measure costs
            if (measureChem) {
                float slice_costCHEM = (Ax_local.array() - bChem.row(s).transpose().array() * 
                                       (Ax_local.array() + eps).log().array()).sum();
                total_costCHEM += slice_costCHEM;
            }
        }
    }
    
    // Apply positivity
    recon.positivity();
    
    // Measure HAADF cost
    if (measureHaadf) {
        total_costHAADF = (g - bh).norm();
    }
    
    return std::make_tuple(total_costHAADF, total_costCHEM);
}

// Multi-GPU Chemical SART
void multigpufusion::chemical_SART(int nIter) {
    #pragma omp parallel num_threads(gpu_ids.size())
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        // Get thread-local GPU context
        FusionGPUContext* ctx = get_thread_context();
        ctx->sart_algo->setGPUIndex(assigned_gpu);
        
        #pragma omp for collapse(2) schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            for (int e = 0; e < Nel; e++) {
                // Copy data to ASTRA (use chemical geometry)
                ctx->vol->copyData((float32*) &recon.data[recon.index(e,s,0,0)]);
                ctx->chem_sino->copyData((float32*) &bChem(s, e*NrowChem));
                
                // SART reconstruction
                int Nproj = NrowChem / Ny;
                ctx->sart_algo->updateSlice(ctx->chem_sino.get(), ctx->vol.get());
                ctx->sart_algo->run(Nproj * nIter);
                
                // Copy back
                memcpy(&recon.data[recon.index(e,s,0,0)], ctx->vol->getData(), sizeof(float)*Ny*Nz);
            }
        }
    }
}

// Multi-GPU Chemical SIRT
void multigpufusion::chemical_SIRT(int nIter) {
    #pragma omp parallel num_threads(gpu_ids.size())
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        // Get thread-local GPU context
        FusionGPUContext* ctx = get_thread_context();
        ctx->sirt_algo->setGPUIndex(assigned_gpu);
        
        #pragma omp for collapse(2) schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            for (int e = 0; e < Nel; e++) {
                // Copy data to ASTRA (use chemical geometry)
                ctx->vol->copyData((float32*) &recon.data[recon.index(e,s,0,0)]);
                ctx->chem_sino->copyData((float32*) &bChem(s, e*NrowChem));
                
                // SIRT reconstruction
                ctx->sirt_algo->updateSlice(ctx->chem_sino.get(), ctx->vol.get());
                ctx->sirt_algo->run(nIter);
                
                // Copy back
                memcpy(&recon.data[recon.index(e,s,0,0)], ctx->vol->getData(), sizeof(float)*Ny*Nz);
            }
        }
    }
}

// Multi-GPU Data Distance
float multigpufusion::data_distance() {
    #pragma omp parallel num_threads(gpu_ids.size())
    {
        int thread_id = omp_get_thread_num();
        int assigned_gpu = gpu_ids[thread_id % gpu_ids.size()];
        
        // Get thread-local GPU context
        FusionGPUContext* ctx = get_thread_context();
        ctx->fp_algo->setGPUIndex(assigned_gpu);
        
        #pragma omp for schedule(dynamic)
        for (int s = 0; s < Nslice; s++) {
            Eigen::VectorXf xx_local(Ny*Nz*Nel);
            
            // Concatenate elements for this slice
            for (int e = 0; e < Nel; e++) {
                memcpy(&xx_local(e*Ny*Nz), &recon.data[recon.index(e,s,0,0)], sizeof(float)*Ny*Nz);
            }
            
            // Forward project all elements (use chemical geometry)
            for (int e = 0; e < Nel; e++) {
                ctx->vol->copyData((float32*) &xx_local(e*Ny*Nz));
                ctx->fp_algo->initialize(cProj, ctx->vol.get(), ctx->chem_sino.get());
                ctx->fp_algo->run();
                memcpy(&gChem(s, e*NrowChem), ctx->chem_sino->getData(), sizeof(float)*NrowChem);
            }
        }
    }
    
    return (gChem - bChem).norm();
}

// Multi-GPU specific utility methods
std::vector<int> multigpufusion::get_gpu_ids() const {
    return gpu_ids;
}

bool multigpufusion::is_multi_gpu_enabled() const {
    return gpu_ids.size() > 1;
}

void multigpufusion::print_gpu_usage() const {
    std::cout << "Active Fusion GPU contexts: " << gpu_contexts.size() << std::endl;
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

//Python functions for multigpufusion module.
PYBIND11_MODULE(multigpufusion, m)
{
    m.doc() = "Multi-GPU Multimodal Fusion Engine using OpenMP";
    py::class_<multigpufusion, multimodal> multigpufusion(m, "multigpufusion");
    multigpufusion.def(py::init<int,int,int>());
    multigpufusion.def(py::init<int,int,int,Vec,Vec>());
    // Multi-GPU versions of key multimodal methods
    multigpufusion.def("poisson_ml", &multigpufusion::poisson_ml, "Multi-GPU Poisson ML Reconstruction");
    multigpufusion.def("data_fusion", &multigpufusion::data_fusion, "Multi-GPU Data Fusion");
    multigpufusion.def("chemical_SART", &multigpufusion::chemical_SART, "Multi-GPU Chemical SART Reconstruction");
    multigpufusion.def("chemical_SIRT", &multigpufusion::chemical_SIRT, "Multi-GPU Chemical SIRT Reconstruction");
    multigpufusion.def("data_distance", &multigpufusion::data_distance, "Multi-GPU Data Distance Calculation");
    // Utility methods
    multigpufusion.def("get_gpu_ids", &multigpufusion::get_gpu_ids, "Get list of GPU IDs being used");
    multigpufusion.def("is_multi_gpu_enabled", &multigpufusion::is_multi_gpu_enabled, "Check if multi-GPU parallelization is enabled");
    multigpufusion.def("print_gpu_usage", &multigpufusion::print_gpu_usage, "Print current GPU usage information");
}