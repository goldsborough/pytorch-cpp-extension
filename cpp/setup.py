import torch.cuda
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension

ext_modules = [
    CppExtension('lltm', ['lltm.cpp']),
]

if torch.cuda.is_available():
    extension = CUDAExtension('lltm_cuda', [
        'lltm_cuda.cpp',
        'lltm_cuda_kernel.cu',
    ])
    ext_modules.append(extension)

setup(
    name='lltm',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': torch.utils.cpp_extension.BuildExtension
    })
