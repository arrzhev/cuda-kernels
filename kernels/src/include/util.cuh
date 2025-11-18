#include <cstdio>

#define cdiv(a, b) ((a + b - 1) / b)

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
    extern __shared__ T smem[];

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
            val = tid < cdiv(blockDim.x, warpSize) ? smem[tid] : 0.0f;
            val = warpReduceSum(val);
        }
    }

    return val;
}