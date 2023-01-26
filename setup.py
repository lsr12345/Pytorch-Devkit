import os
from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def make_cuda_ext(name, module, sources):
    define_macros = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [("WITH_CUDA", None)]
    else:
        raise EnvironmentError('CUDA is required to compile!')

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args={
            'cxx': ['-std=c++14'],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })

#  python setup.py develop
# python setup.py build_ext --inplace
if __name__ == '__main__':
    
    setup(
          name='focalloss',
          version='1.0.0',
          package_data={'tools/loss': ['*/*.so']},
          classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8'
        ],

        ext_modules=[ 
           make_cuda_ext(name='sigmoid_focal_loss_cuda', module='tools.loss',
                  sources=[
                      'src/sigmoid_focal_loss.cpp',
                      'src/sigmoid_focal_loss_cuda.cu'
                  ]),
           make_cuda_ext(name='SigmoidFocalLoss_cuda', module='tools.loss',
                  sources=[
                      'src/SigmoidFocalLoss.cpp',
                      'src/SigmoidFocalLoss_cuda.cu'
                  ])

        ],

        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
