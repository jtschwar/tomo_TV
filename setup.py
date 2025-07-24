from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pybind11
import os

# Get absolute paths to be sure
project_root = os.path.dirname(os.path.abspath(__file__))
eigen_include = os.path.join(project_root, "thirdparty", "eigen")
regularization_include = os.path.join(project_root, "regularization")

ext_modules = [
    Pybind11Extension(
        "multimodal_fusion.ctvlib",
        sources=[
            "regularization/ctvlib.cpp",
            "regularization/bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            regularization_include,
            eigen_include,
        ],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)