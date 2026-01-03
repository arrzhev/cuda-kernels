#ifndef NORMALIZATION_KERNELS
#define NORMALIZATION_KERNELS

#include <util.cuh>

///// Kernels /////

template <unsigned BLOCK_SIZE, bool VEC, InputType T>
__global__ void layerNorm_kernel(const T *X, const T* W, const T *B, T *Y, float eps, unsigned size)
{
    using SumType = float;
    using VecType = typename std::conditional_t<VEC, int4, T>;
    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    const unsigned tx = threadIdx.x;
    const unsigned idx = blockIdx.x * size;

    __shared__ SumType smemM[BLOCK_SIZE];
    __shared__ SumType smemV[BLOCK_SIZE];

    SumType mean = static_cast<SumType>(0);
    SumType var = static_cast<SumType>(0);
    const unsigned sizeV = size / VEC_SIZE;

    for(unsigned i = tx; i < sizeV; i += blockDim.x)
    {
        VecType vecVal = *reinterpret_cast<const VecType*>(&X[idx + i * VEC_SIZE]);
#pragma unroll
        for(unsigned s = 0U; s < VEC_SIZE; ++s)
        {
            SumType val = static_cast<SumType>(reinterpret_cast<const T*>(&vecVal)[s]);
            mean += val;
            var += (val * val);
        }
    }

    smemM[tx] = mean;
    smemV[tx] = var;
    __syncthreads();

    for (unsigned stride = blockDim.x>>1U; stride > 0U; stride>>=1U)
    {
        if (tx < stride)
        {
            smemM[tx] += smemM[tx + stride];
            smemV[tx] += smemV[tx + stride];
        }
        __syncthreads();
    }

    mean = smemM[0U] / size;
    var = rsqrtf((smemV[0U] / size) - (mean * mean) + eps);

    for(unsigned i = tx; i < sizeV; i += blockDim.x)
    {
        VecType vecValX = *reinterpret_cast<const VecType*>(&X[idx + i * VEC_SIZE]);
        VecType vecValW = *reinterpret_cast<const VecType*>(&W[i * VEC_SIZE]);
        VecType vecValB = *reinterpret_cast<const VecType*>(&B[i * VEC_SIZE]);
        T reg[VEC_SIZE];

#pragma unroll
        for(unsigned s = 0U; s < VEC_SIZE; ++s)
        {
            SumType valX = static_cast<SumType>(reinterpret_cast<const T*>(&vecValX)[s]);
            SumType valW = static_cast<SumType>(reinterpret_cast<const T*>(&vecValW)[s]);
            SumType valB = static_cast<SumType>(reinterpret_cast<const T*>(&vecValB)[s]);
            SumType res = ((valX - mean) * var) * valW + valB;
            reg[s] = static_cast<T>(res);
        }

        *reinterpret_cast<VecType*>(&Y[idx + i * VEC_SIZE]) = *reinterpret_cast<VecType*>(&reg);
    }
}

template <unsigned BLOCK_SIZE, bool VEC, InputType T>
__global__ void RMSNorm_kernel(const T *X, const T* W, T *Y, float eps, unsigned size)
{
    using SumType = float;
    using VecType = typename std::conditional_t<VEC, int4, T>;
    constexpr unsigned VEC_SIZE = sizeof(VecType) / sizeof(T);

    const unsigned tx = threadIdx.x;
    const unsigned idx = blockIdx.x * size;

    __shared__ SumType smemV[BLOCK_SIZE];

    SumType var = static_cast<SumType>(0);
    const unsigned sizeV = size / VEC_SIZE;

    for(unsigned i = tx; i < sizeV; i += blockDim.x)
    {
        VecType vecVal = *reinterpret_cast<const VecType*>(&X[idx + i * VEC_SIZE]);
#pragma unroll
        for(unsigned s = 0U; s < VEC_SIZE; ++s)
        {
            SumType val = static_cast<SumType>(reinterpret_cast<const T*>(&vecVal)[s]);
            var += (val * val);
        }
    }

    smemV[tx] = var;
    __syncthreads();

    for (unsigned stride = blockDim.x>>1U; stride > 0U; stride>>=1U)
    {
        if (tx < stride)
            smemV[tx] += smemV[tx + stride];
        __syncthreads();
    }

    var = rsqrtf((smemV[0U] / size) + eps);

    for(unsigned i = tx; i < sizeV; i += blockDim.x)
    {
        VecType vecValX = *reinterpret_cast<const VecType*>(&X[idx + i * VEC_SIZE]);
        VecType vecValW = *reinterpret_cast<const VecType*>(&W[i * VEC_SIZE]);
        T reg[VEC_SIZE];

#pragma unroll
        for(unsigned s = 0U; s < VEC_SIZE; ++s)
        {
            SumType valX = static_cast<SumType>(reinterpret_cast<const T*>(&vecValX)[s]);
            SumType valW = static_cast<SumType>(reinterpret_cast<const T*>(&vecValW)[s]);
            SumType res = valX * var * valW;
            reg[s] = static_cast<T>(res);
        }

        *reinterpret_cast<VecType*>(&Y[idx + i * VEC_SIZE]) = *reinterpret_cast<VecType*>(&reg);
    }
}

///// Launch functions /////

template <unsigned blockDim, InputType T>
void launch_layerNorm(const T *X, const T* W, const T *B, T *Y, float eps, unsigned rows, unsigned cols)
{
    constexpr unsigned VEC_SIZE = sizeof(int4) / sizeof(T);
    const unsigned gridDim = rows;
    //If input tensor is VEC_SIZE aligned - use vectorized memory access
    if(cols % VEC_SIZE == 0U)
    {
        layerNorm_kernel<blockDim, true><<<gridDim, blockDim>>>(X, W, B, Y, eps, cols);
    }
    else
    {
        layerNorm_kernel<blockDim, false><<<gridDim, blockDim>>>(X, W, B, Y, eps, cols);
    }
}

template <unsigned blockDim, InputType T>
void launch_RMSNorm(const T *X, const T* W, T *Y, float eps, unsigned rows, unsigned cols)
{
    constexpr unsigned VEC_SIZE = sizeof(int4) / sizeof(T);
    const unsigned gridDim = rows;
    //If input tensor is VEC_SIZE aligned - use vectorized memory access
    if(cols % VEC_SIZE == 0U)
    {
        RMSNorm_kernel<blockDim, true><<<gridDim, blockDim>>>(X, W, Y, eps, cols);
    }
    else
    {
        RMSNorm_kernel<blockDim, false><<<gridDim, blockDim>>>(X, W, Y, eps, cols);
    }
}

#endif // NORMALIZATION_KERNELS