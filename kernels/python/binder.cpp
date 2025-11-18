#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <tensorAddTorch.hpp>
#include <grayscaleTorch.hpp>
#include <meanBlurTorch.hpp>
#include <tensorMulTorch.hpp>

PYBIND11_MODULE(torch_extension, m) {
  m.def("tensor_add", &tensorAdd, "Cuda tensor addition function");
  m.def("rgb2gray", &rgb2gray, "Cuda rgb to gray image conversion function");
  m.def("mean_blur", &meanBlur, "Cuda mean blur grayscale image function");
  m.def("tensor_mul_naive", &tensorMulNaive, "Cuda naive tensor multiplication function");
  m.def("tensor_mul_shared", &tensorMulShared, "Cuda tensor multiplication function with shared memory optimization");
  m.def("tensor_mul_warp", &tensorMulWarp, "Cuda tensor multiplication function with warp optimization");
}