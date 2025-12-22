#ifndef UTIL_KERNELS
#define UTIL_KERNELS

#include <concepts>
#include <variant>
#include <cstdio>
#include <cuda_fp16.h>

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

template <typename T, typename... AllowedTypes>
concept IsAnyOf = (std::same_as<T, AllowedTypes> || ...);

// Input data type to most of the kernels
template <typename T>
concept InputType = IsAnyOf<T, int, double, float, __half>;

using RuntimeBool = std::variant<std::bool_constant<false>, std::bool_constant<true>>;

inline RuntimeBool to_variant(bool val) {
    return val ? RuntimeBool{std::bool_constant<true>{}} : RuntimeBool{std::bool_constant<false>{}};
}

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
    for (unsigned offset = warpSize >> 1U; offset > 0U; offset>>=1U)
        val += __shfl_down_sync(0xffffffff, val, offset);

    return val;
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T val)
{
    __shared__ T smem[32U]; // max possible size considering 1024 threads per block. 32 = 1024 / 32 (warp size)

    val = warpReduceSum(val);

    if (blockDim.x > warpSize)
    {
        unsigned tid = threadIdx.x;

        unsigned lane = tid % warpSize;
        unsigned wid = tid / warpSize;
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

inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

inline __host__ __device__ float dot(__half a, __half b)
{
    return __half2float(a) * __half2float(b);
}

inline __host__ __device__ float dot(float a, float b)
{
    return a * b;
}

inline __host__ __device__ float dot(float4 a, float4 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

#endif // UTIL_KERNELS