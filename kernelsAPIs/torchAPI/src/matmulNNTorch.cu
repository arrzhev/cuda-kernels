#include <variant>
#include <torch/torch.h>
#include <c10/cuda/CUDAException.h>

#include <matmulNNTorch.hpp>
#include <matmul_nn.cuh>
#include <util.cuh>

#include "common.cuh"

#define CHECK_MATMUL(A, B) CHECK_FPS(A, B); CHECK_CUDA(A); CHECK_CUDA(B); \
        TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "x and y tensors must have dimensions equal to 2"); \
        TORCH_CHECK(A.size(1) == B.size(0), "x and y tensors must have the same cols and rows count");

std::vector<torch::Tensor> matrixMulBias(torch::Tensor A, torch::Tensor B, torch::Tensor bias, bool useReLU, bool transC)
{
    CHECK_MATMUL(A, B);
    CHECK_FP(bias);
    CHECK_CUDA(bias);
    TORCH_CHECK(bias.dim() == 1 && B.size(1) == bias.size(0), "bias dim must be 1D tensor with size equal to B columns");

    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);

    auto C = transC ? torch::empty({N, M}, A.options()).t() : torch::empty({M, N}, A.options());
    auto CR = transC ? torch::empty({N, M}, A.options()).t() : torch::empty({M, N}, A.options());

    const bool AT = A.stride(0) == 1 ? true : false;
    const bool BT = B.stride(0) == 1 ? true : false;
    const bool CT = C.stride(0) == 1 ? true : false;

    const unsigned lda = AT ? M : K;
    const unsigned ldb = BT ? K : N;
    const unsigned ldc = CT ? M : N;
    auto possibleVecAccess = [lda, ldb, ldc](const unsigned vecSize){
        return (lda % vecSize == 0U) && (ldb % vecSize == 0U) && (ldc % vecSize == 0U);
    };

    // Helper function to launch matmul tiles kernel for fp16 data type
    auto launchHalf = [&] <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool VEC, bool ReLU> () {
        launch_matmul_bias_tiles<BM, BN, BK, TM, TN, VEC, ReLU>(
            reinterpret_cast<__half*>(A.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(B.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(bias.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(C.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(CR.data_ptr<torch::Half>()),
            M, N, K, AT, BT, CT);
    };

    // Helper function to launch matmul tiles kernel for fp32 data type
    auto launchFull = [&] <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool VEC, bool ReLU> () {      
        launch_matmul_bias_tiles<BM, BN, BK, TM, TN, VEC, ReLU>(
            A.data_ptr<float>(), B.data_ptr<float>(), bias.data_ptr<float>(), C.data_ptr<float>(), CR.data_ptr<float>(),
                M, N, K, AT, BT, CT);
    };

    // Simplified example of kernel hyper parameters selection based on input data
    if(M > 256U || N > 256U)
    {
        constexpr unsigned BM = 128U;
        constexpr unsigned BN = 128U;
        constexpr unsigned TM = 8U;
        constexpr unsigned TN = 8U;
        
        if(A.scalar_type() == torch::kHalf)
        {
            if(possibleVecAccess(8U))
            {
                constexpr unsigned BK = 16U;
                constexpr bool VEC = true;
                std::visit([&](auto b){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, b.value>();}, to_variant(useReLU));
            }
            else
            {
                constexpr unsigned BK = 8U;
                constexpr bool VEC = false;
                std::visit([&](auto b){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, b.value>();}, to_variant(useReLU));
            }
        }
        else
        {
            constexpr unsigned BK = 16U;
            constexpr bool VEC = true;
            std::visit([&](auto b){launchFull.template operator()<BM, BN, BK, TM, TN, VEC, b.value>();}, to_variant(useReLU));
        }
    }
    else if(M > 128U || N > 128U)
    {
        constexpr unsigned BM = 64U;
        constexpr unsigned BN = 64U;
        constexpr unsigned BK = 16U;

        if(A.scalar_type() == torch::kHalf)
        {
            if(possibleVecAccess(8U))
            {
                constexpr unsigned TM = 8U;
                constexpr unsigned TN = 8U;
                constexpr bool VEC = true;
                std::visit([&](auto b){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, b.value>();}, to_variant(useReLU));
            }
            else
            {
                constexpr unsigned TM = 4U;
                constexpr unsigned TN = 4U;
                constexpr bool VEC = false;
                std::visit([&](auto b){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, b.value>();}, to_variant(useReLU));
            }
        }
        else
        {
            constexpr unsigned TM = 4U;
            constexpr unsigned TN = 4U;
            constexpr bool VEC = true;
            std::visit([&](auto b){launchFull.template operator()<BM, BN, BK, TM, TN, VEC, b.value>();}, to_variant(useReLU));
        }
    }
    else
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        constexpr unsigned TM = 1U;
        constexpr unsigned TN = 1U;
        constexpr bool VEC = false;

        if(A.scalar_type() == torch::kHalf)
            std::visit([&](auto b){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, b.value>();}, to_variant(useReLU));
        else
            std::visit([&](auto b){launchFull.template operator()<BM, BN, BK, TM, TN, VEC, b.value>();}, to_variant(useReLU));
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {C, CR};
}


torch::Tensor matrixMulReLU(torch::Tensor A, torch::Tensor AR, torch::Tensor B, bool transC)
{
    CHECK_MATMUL(A, B);
    CHECK_FP(AR);
    CHECK_CUDA(AR);
    TORCH_CHECK(AR.dim() == A.dim() && AR.size(0) == A.size(0) && AR.size(1) == A.size(1), "pre relu matrix must have same dim and size as A matrix");

    const unsigned M = A.size(0);
    const unsigned K = A.size(1);
    const unsigned N = B.size(1);

    auto C = transC ? torch::empty({N, M}, A.options()).t() : torch::empty({M, N}, A.options());

    const bool AT = A.stride(0) == 1 ? true : false;
    const bool BT = B.stride(0) == 1 ? true : false;
    const bool CT = C.stride(0) == 1 ? true : false;

    const unsigned lda = AT ? M : K;
    const unsigned ldb = BT ? K : N;
    const unsigned ldc = CT ? M : N;
    auto possibleVecAccess = [lda, ldb, ldc](const unsigned vecSize){
        return (lda % vecSize == 0U) && (ldb % vecSize == 0U) && (ldc % vecSize == 0U);
    };

    // Lambda function to select 'split K' number based on mat size
    auto getSK = [&](){
        using VarSK = std::variant<std::integral_constant<unsigned, 4U>, std::integral_constant<unsigned, 1U>>;
        return K > 2U * std::max(M, N) ? VarSK{std::integral_constant<unsigned, 4U>{}} : VarSK{std::integral_constant<unsigned, 1U>{}};
    };

    // Helper function to launch matmul tiles kernel for fp16 data type
    auto launchHalf = [&] <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool VEC, unsigned SK> () {
        if(SK > 1U)
            C.zero_();  // Zero initialization required for 'split K' optimization

        launch_matmul_relu_tiles<BM, BN, BK, TM, TN, VEC, SK>(
            reinterpret_cast<__half*>(A.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(AR.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(B.data_ptr<torch::Half>()),
            reinterpret_cast<__half*>(C.data_ptr<torch::Half>()),
            M, N, K, AT, BT, CT);
    };

    // Helper function to launch matmul tiles kernel for fp32 data type
    auto launchFull = [&] <unsigned BM, unsigned BN, unsigned BK, unsigned TM, unsigned TN, bool VEC, unsigned SK> () {
        if(SK > 1U)
            C.zero_(); // Zero initialization required for 'split K' optimization
        
        launch_matmul_relu_tiles<BM, BN, BK, TM, TN, VEC, SK>(
            A.data_ptr<float>(), AR.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
                M, N, K, AT, BT, CT);
    };

    // Simplified example of kernel hyper parameters selection based on input data
    if(M > 256U || N > 256U)
    {
        constexpr unsigned BM = 128U;
        constexpr unsigned BN = 128U;
        constexpr unsigned TM = 8U;
        constexpr unsigned TN = 8U;
        
        if(A.scalar_type() == torch::kHalf)
        {
            if(possibleVecAccess(8U))
            {
                constexpr unsigned BK = 16U;
                constexpr bool VEC = true;
                std::visit([&](auto sk){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, sk.value>();}, getSK());
            }
            else
            {
                constexpr unsigned BK = 8U;
                constexpr bool VEC = false;
                std::visit([&](auto sk){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, sk.value>();}, getSK());
            }
        }
        else
        {
            constexpr unsigned BK = 8U;
            constexpr bool VEC = true;
            std::visit([&](auto sk){launchFull.template operator()<BM, BN, BK, TM, TN, VEC, sk.value>();}, getSK());
        }
    }
    else if(M > 128U || N > 128U)
    {
        constexpr unsigned BM = 64U;
        constexpr unsigned BN = 64U;
        constexpr unsigned BK = 16U;

        if(A.scalar_type() == torch::kHalf)
        {
            if(possibleVecAccess(8U))
            {
                constexpr unsigned TM = 8U;
                constexpr unsigned TN = 8U;
                constexpr bool VEC = true;
                std::visit([&](auto sk){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, sk.value>();}, getSK());
            }
            else
            {
                constexpr unsigned TM = 4U;
                constexpr unsigned TN = 4U;
                constexpr bool VEC = false;
                std::visit([&](auto sk){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, sk.value>();}, getSK());
            }
        }
        else
        {
            constexpr unsigned TM = 4U;
            constexpr unsigned TN = 4U;
            constexpr bool VEC = false;
            std::visit([&](auto sk){launchFull.template operator()<BM, BN, BK, TM, TN, VEC, sk.value>();}, getSK());
        }
    }
    else
    {
        constexpr unsigned BM = 32U;
        constexpr unsigned BN = 32U;
        constexpr unsigned BK = 32U;
        constexpr unsigned TM = 1U;
        constexpr unsigned TN = 1U;
        constexpr bool VEC = false;

        if(A.scalar_type() == torch::kHalf)
            std::visit([&](auto sk){launchHalf.template operator()<BM, BN, BK, TM, TN, VEC, sk.value>();}, getSK());
        else
            std::visit([&](auto sk){launchFull.template operator()<BM, BN, BK, TM, TN, VEC, sk.value>();}, getSK());
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}
