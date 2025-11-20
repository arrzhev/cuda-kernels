#ifndef UTIL_KERNELS
#define UTIL_KERNELS

#include <cstdio>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define cudaCheckErrors(func) { gpuAssert(func, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaError::cudaSuccess) 
   {
      fprintf(stderr,"GPU fatal error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
      if(abort)
        exit(1);
   }
}

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val)
{
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    return val;
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T val)
{
    __shared__ T smem[32]; // max possible size considering 1024 threads per block. 32 = 1024 / 32 (warp size)

    val = warpReduceSum(val);

    if (blockDim.x > warpSize)
    {
        int tid = threadIdx.x;

        int lane = tid % warpSize;
        int wid = tid / warpSize;
        if (lane == 0)
            smem[wid] = val;

        __syncthreads();

        if (tid < warpSize)
        {
            val = tid < CEIL_DIV(blockDim.x, warpSize) ? smem[tid] : 0.0f;
            val = warpReduceSum(val);
        }
    }

    return val;
}

#endif // UTIL_KERNELS