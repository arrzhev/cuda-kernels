#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tensorAddTorch.hpp>
#include <grayscaleTorch.hpp>
#include <meanBlurTorch.hpp>
#include <vectorDotProductTorch.hpp>
#include <matrixVectorMulTorch.hpp>
#include <matmulTorch.hpp>
#include <tensorMulTorch.hpp>

PYBIND11_MODULE(torch_extension, m) 
{
    m.def("tensor_add", &tensorAdd, "Cuda tensor addition function");

    m.def("rgb2gray", &rgb2gray, "Cuda rgb to gray image conversion function");

    m.def("mean_blur", &meanBlur, "Cuda mean blur grayscale image function");

    m.def("vector_dot_product", &vectorDotProduct, "Cuda vector dot product function");

    m.def("matrix_vector_mul", &matrixVectorMul, "Cuda matrix vector multiplication function");
    m.def("matrix_vector_mul_naive", &matrixVectorMulNaive, "Cuda naive matrix vector multiplication function");
    m.def("matrix_vector_mul_shared", &matrixVectorMulShared, "Cuda matrix vector multiplication function with shared memory optimization");
    m.def("matrix_vector_mul_warp", &matrixVectorMulWarp, "Cuda matrix vector multiplication function with warp optimization");

    m.def("matmul", &matrixMul, "Cuda matmul Optimized function");
    m.def("matmul_naive", &matmulNaive, "Cuda matmul Naive function");
    m.def("matmul_naive_K", &matmulNaiveK, "Cuda matmul Naive function with K split");
    m.def("matmul_coalescing", &matmulCoalescing, "Cuda matmul function with Coalescing memory optimization");
    m.def("matmul_coalescing_K", &matmulCoalescingK, "Cuda matmul function with Coalescing memory and split K optimizations");
    m.def("matmul_BTiles", &matmulBTiles, "Cuda matmul function with Block tiles memory optimization");
    m.def("matmul_BTiles_K", &matmulBTilesK, "Cuda matmul function with Block tiles and split K memory optimization");
    m.def("matmul_BTiles_DBuf", &matmulBTilesDBuf, "Cuda matmul function with Block tiles and Double buffer memory optimizations");
    m.def("matmul_TTiles_1D", &matmulTTiles1D, "Cuda matmul function with Thread tiles 1D memory optimization");
    m.def("matmul_TTiles_1D_K", &matmulTTiles1DK, "Cuda matmul function with Thread tiles 1D and split K memory optimization");
    m.def("matmul_TTiles_1D_DBuf", &matmulTTiles1DDBuf, "Cuda matmul function with Thread tiles 1D and Double buffer memory optimizations");
    m.def("matmul_TTiles_2D", &matmulTTiles2D, "Cuda matmul function with Thread tiles 2D memory optimization");
    m.def("matmul_TTiles_2D_K", &matmulTTiles2DK, "Cuda matmul function with Thread tiles 2D and split K memory optimization");
    m.def("matmul_TTiles_2D_DBuf", &matmulTTiles2DDBuf, "Cuda matmul function with Thread tiles 2D and Double buffer memory optimizations");
    m.def("matmul_TTiles_2D_vec", &matmulTTiles2DVec, "Cuda matmul function with Vectorized Thread tiles 2D memory optimization");
    m.def("matmul_TTiles_2D_vec_K", &matmulTTiles2DVecK, "Cuda matmul function with Vectorized Thread tiles 2D and split K memory optimization");
    m.def("matmul_TTiles_2D_DBuf_vec", &matmulTTiles2DDBufVec, "Cuda matmul function with Vectorized Thread tiles 2D and Double buffer memory optimization");

    m.def("tensor_mul", &tensorMul, "Cuda tensor multiplication function");
}