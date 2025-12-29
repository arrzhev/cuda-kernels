#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tensorAddTorch.hpp>
#include <grayscaleTorch.hpp>
#include <meanBlurTorch.hpp>
#include <vectorDotProductTorch.hpp>
#include <matrixVectorMulTorch.hpp>
#include <matmulTorch.hpp>
#include <matmulNNTorch.hpp>
#include <tensorMulTorch.hpp>
#include <matrixReductionTorch.hpp>
#include <layerNormTorch.hpp>

using namespace pybind11::literals;

PYBIND11_MODULE(torch_extension, m) 
{
    // Tensor Add
    m.def("tensor_add", &tensorAdd, "Cuda tensor addition function");

    // Grayscale
    m.def("rgb2gray", &rgb2gray, "Cuda rgb to gray image conversion function");

    // Blur
    m.def("mean_blur", &meanBlur, "Cuda mean blur grayscale image function");

    // Dot product
    m.def("vector_dot_product", &vectorDotProduct, "Cuda vector dot product function");

    // Matrix x Vector
    m.def("matrix_vector_mul", &matrixVectorMul, "Cuda matrix vector multiplication function");
    m.def("matrix_vector_mul_naive", &matrixVectorMulNaive, "Cuda naive matrix vector multiplication function");
    m.def("matrix_vector_mul_shared", &matrixVectorMulShared, "Cuda matrix vector multiplication function with shared memory optimization");
    m.def("matrix_vector_mul_warp", &matrixVectorMulWarp, "Cuda matrix vector multiplication function with warp optimization");

    // Matmul
    m.def("matmul", &matrixMul, "Cuda matmul Optimized function",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_naive", &matmulNaive, "Cuda matmul Naive function",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_naive_K", &matmulNaiveK, "Cuda matmul Naive function with K split",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_coalescing", &matmulCoalescing, "Cuda matmul function with Coalescing memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_coalescing_K", &matmulCoalescingK, "Cuda matmul function with Coalescing memory and split K optimizations",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_BTiles", &matmulBTiles, "Cuda matmul function with Block tiles memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_BTiles_K", &matmulBTilesK, "Cuda matmul function with Block tiles and split K memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_BTiles_DBuf", &matmulBTilesDBuf, "Cuda matmul function with Block tiles and Double buffer memory optimizations",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_BTiles_DBuf_K", &matmulBTilesDBufK, "Cuda matmul function with Block tiles, Double buffer and split K memory optimizations",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_1D", &matmulTTiles1D, "Cuda matmul function with Thread tiles 1D memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_1D_K", &matmulTTiles1DK, "Cuda matmul function with Thread tiles 1D and split K memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_1D_DBuf", &matmulTTiles1DDBuf, "Cuda matmul function with Thread tiles 1D and Double buffer memory optimizations",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_1D_DBuf_K", &matmulTTiles1DDBufK, "Cuda matmul function with Thread tiles 1D, Double buffer and split K memory optimizations",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_2D", &matmulTTiles2D, "Cuda matmul function with Thread tiles 2D memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_2D_K", &matmulTTiles2DK, "Cuda matmul function with Thread tiles 2D and split K memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_2D_DBuf", &matmulTTiles2DDBuf, "Cuda matmul function with Thread tiles 2D and Double buffer memory optimizations",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_2D_DBuf_K", &matmulTTiles2DDBufK, "Cuda matmul function with Thread tiles 2D, Double buffer and split K memory optimizations",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_2D_vec", &matmulTTiles2DVec, "Cuda matmul function with Vectorized Thread tiles 2D memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_2D_vec_K", &matmulTTiles2DVecK, "Cuda matmul function with Vectorized Thread tiles 2D and split K memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_2D_DBuf_vec", &matmulTTiles2DDBufVec, "Cuda matmul function with Vectorized Thread tiles 2D and Double buffer memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_TTiles_2D_DBuf_vec_K", &matmulTTiles2DDBufVecK, "Cuda matmul function with Vectorized Thread tiles 2D, Double buffer and split K memory optimization",
         "A"_a, "B"_a, "transC"_a = false);
    m.def("matmul_BTiles_vec_wmma", &matmulBTilesVecWMMA, "Cuda matmul function with Vectorized Block tiles and tensor cores optimization",
         "A"_a, "B"_a, "transC"_a = false);

    // Tensor mul
    m.def("tensor_mul", &tensorMul, "Cuda tensor multiplication function");

    // Matmul NN
    m.def("matmul_bias", &matrixMulBias, "Cuda matmul with bias Optimized function",
         "A"_a, "B"_a, "bias"_a, "use_relu"_a = false, "transC"_a = false);

    m.def("matmul_relu", &matrixMulReLU, "Cuda matmul with bias Optimized function",
         "A"_a, "AR"_a, "B"_a, "transC"_a = false);

    // Matrix reduction
    m.def("matrix_reduction_row", &matrixRowReduction, "Cuda matrix row reduction function");
    m.def("matrix_reduction_row_relu", &matrixRowReductionReLU, "Cuda matrix row reduction fused with ReLU backward step function");

    // Layer norm
    m.def("layer_norm", &layerNorm, "Cuda layer normalization function",
           "X"_a, "W"_a, "B"_a, "eps"_a = 1e-5);
}