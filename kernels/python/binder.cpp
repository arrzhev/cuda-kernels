#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tensorAddTorch.hpp>
#include <grayscaleTorch.hpp>
#include <meanBlurTorch.hpp>
#include <vectorDotProductTorch.hpp>
#include <matrixVectorMulTorch.hpp>
#include <matrixMulTorch.hpp>
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

    m.def("matrix_mul", &matrixMul, "Cuda matrix multiplication function");
    m.def("matrix_mul_naive", &matrixMulNaive, "Cuda naive matrix multiplication function");
    m.def("matrix_mul_coalescing", &matrixMulCoalescing, "Cuda matrix multiplication function with coalescing memory optimization");
    m.def("matrix_mul_tiled", &matrixMulTiled, "Cuda matrix multiplication function with tiled memory optimization");
    m.def("matrix_mul_tiled_1D", &matrixMulTiled1D, "Cuda matrix multiplication function with 1D tiled memory optimization");
    m.def("matrix_mul_tiled_2D", &matrixMulTiled2D, "Cuda matrix multiplication function with 2D tiled memory optimization");
    m.def("matrix_mul_tiled4_2D", &matrixMulTiled2D4, "Cuda matrix multiplication function with vectorized 2D tiled memory optimization");

    m.def("tensor_mul", &tensorMul, "Cuda tensor multiplication function");
}