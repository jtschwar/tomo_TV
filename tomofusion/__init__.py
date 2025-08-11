"""
tomo_TV: Python and C++ toolbox for tomographic data processing 
and developing iterative reconstruction algorithms.
"""

__version__ = "0.1.0"
__author__ = "Jonathan Schwartz"
__email__ = "jtschw@umich.edu"

def device_count():
    """Detect available GPUs using CUDA"""
    try:
        import pycuda.driver as cuda
        cuda.init()
        gpu_count = cuda.Device.count()
        return list(range(gpu_count))
    except:
        return []


def determine_gpu_config(gpu_id: int = -1):
    """
    Determine the GPU configuration based on the available GPUs and user input.
    """

    ngpus = len(device_count())
    if ngpus == 0:
        raise ValueError('An Nvidia GPU is Needed for this package!')
    if gpu_id < 0 and ngpus == 1:
        return 'singleconfig'
    elif gpu_id < 0 and ngpus > 1:
        return 'multigpu'
    else: # Here assume user wants to specify gpu
        return 'singleconfig'
