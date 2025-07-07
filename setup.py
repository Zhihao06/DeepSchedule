import os
import subprocess
import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == '__main__':
    nvshmem_dir = os.getenv('NVSHMEM_DIR', None)
    assert nvshmem_dir is not None and os.path.exists(nvshmem_dir), 'Failed to find NVSHMEM'
    print(f'NVSHMEM directory: {nvshmem_dir}')
    cuda_dir = os.getenv('CUDA_HOME', "/usr/local/cuda")

    # TODO: currently, we only support Hopper architecture, we may add Ampere support later
    os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
    cxx_flags = ['-O3', '-Wno-deprecated-declarations', '-Wno-unused-variable',
                 '-Wno-sign-compare', '-Wno-reorder', '-Wno-attributes']
    nvcc_flags = ['-O3', '-Xcompiler', '-O3', '-rdc=true', '--ptxas-options=--register-usage-level=10']
    include_dirs = [os.path.abspath('deep_ep/csrc'), os.path.abspath('deep_gemm/include'), os.path.abspath(f'{cuda_dir}/include'), os.path.abspath(f'{nvshmem_dir}/include')]
    sources = ['fuse_kernel/src/deep_fuse.cpp']
    library_dirs = [os.path.abspath('deep_ep/csrc/build'), os.path.abspath('deep_gemm/kernels/build'), os.path.abspath('fuse_kernel/build/src/ops')]
    extra_link_args = [
        '-ldeep_gemm_kernels', 
        f'-Wl,-rpath,{os.path.abspath("deep_ep/csrc/build")},-ldeep_ep', 
        f'-Wl,-rpath,{os.path.abspath("fuse_kernel/build/src/ops")},-lfuse_silu_and_mul'
    ]

    # Disable aggressive PTX instructions
    if int(os.getenv('DISABLE_AGGRESSIVE_PTX_INSTRS', '0')):
        cxx_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')
        nvcc_flags.append('-DDISABLE_AGGRESSIVE_PTX_INSTRS')

    cxx_flags.append('-DUSE_C10D_NCCL')

    # Disable DLTO (default by PyTorch)
    # nvcc_dlink = ['-dlink', f'-L{nvshmem_dir}/lib', '-lnvshmem']
    # extra_link_args += ['-l:libnvshmem.a', '-l:nvshmem_bootstrap_uid.so', f'-Wl,-rpath,{nvshmem_dir}/lib']
    extra_compile_args = {
        'cxx': cxx_flags,
        'nvcc': nvcc_flags
    }

    # noinspection PyBroadException
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        revision = ''

    setuptools.setup(
        name='deep_fuse',
        version='1.0.0' + revision,
        ext_modules=[
            CUDAExtension(
                name='deep_fuse_cpp',
                include_dirs=include_dirs,
                library_dirs=library_dirs,
                sources=sources,
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
