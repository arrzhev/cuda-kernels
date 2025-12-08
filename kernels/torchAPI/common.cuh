#ifndef COMMON_TORCH
#define COMMON_TORCH

#define CHECK_FP16(x) TORCH_CHECK((x).scalar_type() == torch::kHalf, #x " must be a FP16 tensor")
#define CHECK_FP32(x) TORCH_CHECK((x).scalar_type() == torch::kFloat, #x " must be a FP32 tensor")

#define CHECK_UINT8(x) TORCH_CHECK((x).scalar_type() == torch::kUInt8, #x " must be a UINT8 tensor")

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#endif // COMMON_TORCH