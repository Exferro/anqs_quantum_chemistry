import torch as pt
try:
    import cupy
except ImportError:
    print("CUPY cannot initialize, not using CUDA kernels")


class CUDAStream:
    ptr = pt.cuda.current_stream().cuda_stream


cuda_int64popcount_code = '''
    extern "C" __global__ void cuda_int64popcount(
        const int n,
        const long long int* a,
        long long int* dist
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        dist[elem_idx] = __popcll(a[elem_idx]);
    }
'''

cuda_int64popcount_kernel = cupy.RawKernel(cuda_int64popcount_code,
                                           'cuda_int64popcount',
                                           options=('--std=c++11',),
                                           backend='nvrtc')


def cuda_int64_popcount(a):
    if not a.is_contiguous():
        a.contiguous()
    assert (a.dtype == pt.int64)

    popcount = pt.zeros_like(a)
    n = popcount.nelement()

    cuda_int64popcount_kernel(grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                              block=tuple([512, 1, 1]),
                              args=(n,
                                    a.data_ptr(),
                                    popcount.data_ptr()),
                              stream=CUDAStream)

    return popcount


cuda_int64popcount__code = '''
    extern "C" __global__ void cuda_int64popcount_(
        const int n,
        long long int* a
    ) {
        int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (elem_idx >= n) {
            return;
        }

        a[elem_idx] = __popcll(a[elem_idx]);
    }
'''

cuda_int64popcount__kernel = cupy.RawKernel(cuda_int64popcount__code,
                                            'cuda_int64popcount_',
                                            options=('--std=c++11',),
                                            backend='nvrtc')

#@cupy._util.memoize(for_each_device=True)

def cuda_int64_popcount_(a):
    if not a.is_contiguous():
        a.contiguous()
    assert (a.dtype == pt.int64)

    n = a.nelement()

    cuda_int64popcount__kernel(grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                              block=tuple([512, 1, 1]),
                              args=(n,
                                    a.data_ptr()),
                              stream=CUDAStream)

    return a
