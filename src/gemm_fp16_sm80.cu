#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#define MAKE_GEMM_4D_TEST(batch_, M_, N_, K_, RowMajorA, RowMajorB, RowMajorC) std::make_tuple(batch_, M_, N_, K_, RowMajorA, RowMajorB, RowMajorC)

/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                          \
    {                                                                                                  \
        cutlass::Status error = status;                                                                \
        if (error != cutlass::Status::kSuccess)                                                        \
        {                                                                                              \
            std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                      << std::endl;                                                                    \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                                    \
    {                                                                         \
        cudaError_t error = status;                                           \
        if (error != cudaSuccess)                                             \
        {                                                                     \
            std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                      << " at line: " << __LINE__ << std::endl;               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

template <bool RowMajorA, bool RowMajorB, bool RowMajorC>
void cpugemm(half *A, half *B, half *C, int M, int N, int K, int batch)
{
    for (int i = 0; i < batch; i++)
    {
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                float res = 0.f;
                for (int k = 0; k < K; k++)
                {
                    half a = RowMajorA ? A[i * M * K + m * K + k] : A[i * M * K + k * M + m];
                    half b = RowMajorB ? B[i * N * K + k * N + n] : B[i * N * K + n * K + k];
                    res += __half2float(a) * __half2float(b);
                }
                unsigned ci = RowMajorC ? (i * M * N + m * N + n) : (i * M * N + n * M + m);
                C[ci] = __float2half(res);
            }
        }
    }
}

template <bool RowMajorA, bool RowMajorB, bool RowMajorC>
bool verify(half *h_A, half *h_B, half *h_C, int M, int N, int K, int batch, float eps)
{
    bool correct = true;
    int mem_size_C = M * N * sizeof(half) * batch;
    half *realC = (half *)malloc(mem_size_C);
    cpugemm<RowMajorA, RowMajorB, RowMajorC>(h_A, h_B, realC, M, N, K, batch);

    for (int i = 0; i < static_cast<int>(M * N); i++)
    {
        float abs_err = fabs((float)h_C[i] - (float)realC[i]);
        float abs_val = fabs((float)h_C[i]);
        float rel_err = abs_err / (abs_val * K);

        if (rel_err > eps || std::isnan((float)realC[i]) || std::isnan((float)h_C[i]))
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %lf\n", i, (float)h_C[i], (float)realC[i], eps);
            correct = false;
            break;
        }
    }
    free(realC);
    return correct;
}

void RandInit(half *data, unsigned size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-6.0, 6.0);

    for (unsigned i = 0; i < size; ++i)
    {
        data[i] = half(dist(gen));
    }
}

template <bool RowMajorA, bool RowMajorB, bool RowMajorC>
void TestCutlassSgemm(int batch, int M, int N, int K, const half *d_A, const half *d_B, half *d_C)
{
    // The code section below describes datatype for input, output matrices and computation between
    // elements in input matrices.
    using ElementAccumulator = float;                  // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;             // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;             // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;             // <- data type of elements in output matrix D

    // The code section below describes matrix layout of input and output matrices. Column Major for
    // Matrix A, Row Major for Matrix B and Row Major for Matrix C
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
    using MMAOp = cutlass::arch::OpClassTensorOp;

    // This code section describes CUDA SM architecture number
    using SmArch = cutlass::arch::Sm80;

    // This code section describes the tile size a thread block will compute
    using ShapeMMAThreadBlock =
        cutlass::gemm::GemmShape<128, 128, 32>; // <- threadblock tile M = 128, N = 128, K = 32
    // This code section describes tile size a warp will compute
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>; // <- warp tile M = 64, N = 64, K = 32
    // This code section describes the size of MMA op
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>; // <- MMA Op tile M = 8, N = 8, K = 4

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

    // This code section describes ?
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                    // <- data type of output matrix
        128 / cutlass::sizeof_bits<ElementOutput>::value, // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
        ElementAccumulator,                               // <- data type of accumulator
        ElementComputeEpilogue>;                          // <- data type for alpha/beta in linear combination function

    // Number of pipelines you want to use
    constexpr int NumStages = 2;

    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                             LayoutInputA,
                                             ElementInputB,
                                             LayoutInputB,
                                             ElementOutput,
                                             LayoutOutput,
                                             ElementAccumulator,
                                             MMAOp,
                                             SmArch,
                                             ShapeMMAThreadBlock,
                                             ShapeMMAWarp,
                                             ShapeMMAOp,
                                             EpilogueOp,
                                             SwizzleThreadBlock,
                                             NumStages>;

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    typename Gemm::Arguments arguments{problem_size,                                    // <- problem size of matrix multiplication
                                       {reinterpret_cast<const cutlass::half_t *>(d_A), lda}, // <- reference to matrix A on device
                                       {reinterpret_cast<const cutlass::half_t *>(d_B), ldb}, // <- reference to matrix B on device
                                       {reinterpret_cast<cutlass::half_t *>(d_C), ldc}, // <- reference to matrix C on device
                                       {reinterpret_cast<cutlass::half_t *>(d_C), ldc}, // <- reference to matrix D on device
                                       {alpha, beta},                                   // <- tuple of alpha and beta
                                       split_k_slices};                                 // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
template <bool RowMajorA, bool RowMajorB, bool RowMajorC>
int MatrixMultiply(int batch, unsigned M, unsigned N, unsigned K, bool verifyResult)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = M * K * batch;
    unsigned int mem_size_A = sizeof(half) * size_A;
    half *h_A = new half[size_A]{};

    unsigned int size_B = K * N * batch;
    unsigned int mem_size_B = sizeof(half) * size_B;
    half *h_B = new half[size_B]{};

    RandInit(h_A, size_A);
    RandInit(h_B, size_B);

    // Allocate device memory
    half *d_A, *d_B, *d_C;

    // Allocate host matrix C
    unsigned int size_C = M * N * batch;
    unsigned int mem_size_C = sizeof(half) * size_C;
    half *h_C = new half[size_C]{};

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));
    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // copy host memory to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // Performs warmup operation using matrixMul CUDA kernel
    TestCutlassSgemm<RowMajorA, RowMajorB, RowMajorC>(batch, M, N, K, d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start));

    // Execute the kernel
    int nIter = 10;
    for (int j = 0; j < nIter; j++)
    {
        TestCutlassSgemm<RowMajorA, RowMajorB, RowMajorC>(batch, M, N, K, d_A, d_B, d_C);
    }

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop));
    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    double msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) * batch;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    double bandWidth = (double)(M * K + N * K + M * N) * 2.0 / (msecPerMatrixMul * 1000.0 * 1000.0);
    printf("%.2f GFlop/s, %.3f ms, %.2f GB/s(Memory)\n", gigaFlops, msecPerMatrixMul, bandWidth);

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    bool correct = true;
    if (verifyResult)
    {
        printf("Checking computed result for correctness: ");
        bool correct = verify<RowMajorA, RowMajorB, RowMajorC>(h_A, h_B, h_C, M, N, K, batch, 1e-2);
        printf("%s\n\n", correct ? "Result = PASS" : "Result = FAIL");
    }

    // Clean up memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    if (correct)
    {
        return EXIT_SUCCESS;
    }
    else
    {
        return EXIT_FAILURE;
    }
}

/**
 * Program main
 */
int main(int argc, char **argv)
{
    bool verifyResult = true;
    int matrix_result;

    std::vector<std::tuple<int, int, int, int, bool, bool, bool>> test_datas = {
        MAKE_GEMM_4D_TEST(1, 256, 256, 256, true, true, true),
        MAKE_GEMM_4D_TEST(1, 512, 512, 512, true, true, true),
        MAKE_GEMM_4D_TEST(1, 1024, 1024, 1024, true, true, true),
        MAKE_GEMM_4D_TEST(1, 2048, 2048, 2048, true, true, true),
    };

    for (int i = 0; i < test_datas.size(); i++)
    {
        auto [batch, M, N, K, RowMajorA, RowMajorB, RowMajorC] = test_datas[i];

        printf("batch:%d, M:%d, N:%d, K:%d RowMajorA:%d RowMajorB:%d RowMajorC:%d \n", batch, M, N, K, RowMajorA, RowMajorB, RowMajorC);
        matrix_result = MatrixMultiply<true, false, true>(batch, M, N, K, verifyResult);
    }

    exit(matrix_result);
}