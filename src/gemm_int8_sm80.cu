#include <algorithm>
#include <cfloat>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
using ElementInputA = int8_t;                      // <- data type of elements in input matrix A
using ElementInputB = int8_t;                      // <- data type of elements in input matrix B
using ElementOutput = int32_t;                      // <- data type of elements in output matrix D

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
    cutlass::gemm::GemmShape<128, 256, 64>; // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>; // <- warp tile M = 64, N = 64, K = 64
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 16>; // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                    // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
                                                      // memory access. For a byte, it's 16
                                                      // elements. This becomes the vector width of
                                                      // math instructions in the epilogue too
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

#define MAKE_GEMM_4D_TEST(batch_, M_, N_, K_) std::make_tuple(batch_, M_, N_, K_)

void RandInit(int8_t *data, int size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(-20, 20);

    for (int i = 0; i < size; ++i)
    {
        data[i] = (int8_t)dist(gen);
    }
}

template <bool NoTransA, bool NoTransB, bool RowMajorC>
void cpugemm(int8_t *A, int8_t *B, int32_t *C, int M, int N, int K)
{
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            int32_t res = 0;
            for (int k = 0; k < K; k++)
            {
                int8_t a = NoTransA ? A[m * K + k] : A[k * M + m];
                int8_t b = NoTransB ? B[k * N + n] : B[n * K + k];
                res += (int)a * (int)b;
            }
            unsigned ci = RowMajorC ? (m * N + n) : (n * M + m);
            C[ci] = res;
        }
    }
}

template <bool NoTransA, bool NoTransB, bool RowMajorC>
bool verify(int8_t *h_A, int8_t *h_B, int32_t *h_C, int M, int N, int K)
{
    bool correct = true;
    int mem_size_C = M * N * sizeof(int);
    int32_t *realC = (int32_t *)malloc(mem_size_C);
    memset(realC, 0, mem_size_C);
    cpugemm<NoTransA, NoTransB, RowMajorC>(h_A, h_B, realC, M, N, K);

    for (int i = 0; i < static_cast<int>(M * N); i++)
    {
        if ((int)h_C[i] != (int)realC[i])
        {
            printf("Error! Matrix[%05d]=%d, ref=%d\n", i, (int)h_C[i], realC[i]);
            correct = false;
            break;
        }
    }
    free(realC);
    return correct;
}

template <
    bool NoTransA,
    bool NoTransB,
    bool RowMajorC>
void TestCutlassGemm(int8_t *d_A, int8_t *d_B, int32_t *d_C, unsigned M, unsigned N, unsigned K)
{
    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Create a tuple of problem size for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
    // instantiated CUTLASS kernel
    typename Gemm::Arguments arguments{problem_size,    // <- problem size of matrix multiplication
                                       {d_A, lda},      // <- reference to matrix A on device
                                       {d_B, ldb},      // <- reference to matrix B on device
                                       {d_C, ldc},      // <- reference to matrix C on device
                                       {d_C, ldc},      // <- reference to matrix D on device
                                       {alpha, beta},   // <- tuple of alpha and beta
                                       split_k_slices}; // <- k-dimension split factor

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

template <bool NoTransA, bool NoTransB, bool RowMajorC>
int MatrixMultiply(
    unsigned M, unsigned N, unsigned K, bool verifyResult, int batch, int repeat_number)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = M * K * batch;
    unsigned int mem_size_A = sizeof(int8_t) * size_A;
    int8_t *h_A = new int8_t[size_A];

    unsigned int size_B = K * N * batch;
    unsigned int mem_size_B = sizeof(int8_t) * size_B;
    int8_t *h_B = new int8_t[size_B];

    RandInit(h_A, size_A);
    RandInit(h_B, size_B);

    // Allocate host matrix C
    unsigned int size_C = M * N * batch;
    unsigned int mem_size_C = M * N * sizeof(int32_t) * batch;
    int32_t *h_C = new int32_t[size_C];

    // Allocate device memory
    int8_t *d_A, *d_B;
    int32_t *d_C;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    // copy host memory to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Performs warmup operation using matrixMul CUDA kernel
    TestCutlassGemm<true, true, true>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start));

    // Execute the kernel
    int nIter = repeat_number;
    for (int j = 0; j < nIter; j++)
    {
        TestCutlassGemm<true, true, true>(d_A, d_B, d_C, M, N, K);
    }

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    double msecPerMatrixMul = (double)msecTotal / nIter;
    double flopsPerMatrixMul =
        2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) * batch;
    double gigaFlops = (flopsPerMatrixMul) / (msecPerMatrixMul * 1000.0 * 1000.0);
    double bandWidth = (double)(M * K + N * K + M * N * 4) / (msecPerMatrixMul * 1000 * 1000);

    printf("%.2f GFlop/s, %.3f ms, %.2f GB/s(Memory)\n", (float)gigaFlops, (float)msecPerMatrixMul, (float)bandWidth);

    bool correct = true;
    if (verifyResult)
    {
        printf("Checking computed result for correctness: ");
        bool correct = verify<NoTransA, NoTransB, RowMajorC>(h_A, h_B, h_C, M, N, K);
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

    int repeat_number = 10;

    std::vector<std::tuple<int, int, int, int>> test_datas = {
        MAKE_GEMM_4D_TEST(1, 256, 256, 256),
        MAKE_GEMM_4D_TEST(1, 512, 512, 512),
        MAKE_GEMM_4D_TEST(1, 1024, 1024, 1024),
        MAKE_GEMM_4D_TEST(1, 2048, 2048, 2048),
    };

    for (int i = 0; i < test_datas.size(); i++)
    {
        auto [batch, M, N, K] = test_datas[i];

        printf("batch:%d, M:%d, N:%d, K:%d \n", batch, M, N, K);
        matrix_result = MatrixMultiply<true, false, true>(M, N, K, verifyResult, batch, repeat_number);
    }

    exit(matrix_result);
}