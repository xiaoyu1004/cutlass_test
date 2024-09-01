#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <iostream>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

#include "cutlass/gemm/device/gemm.h"

#define MAKE_GEMM_4D_TEST(batch_, M_, N_, K_, RowMajorA, RowMajorB, RowMajorC) std::make_tuple(batch_, M_, N_, K_, RowMajorA, RowMajorB, RowMajorC)

#define PRINT_MATRIX 0

#define CUDA_CHECK(func)                                                           \
    {                                                                              \
        cudaError_t e = (func);                                                    \
        if (e != cudaSuccess)                                                      \
        {                                                                          \
            printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        }                                                                          \
    }

template <bool RowMajorA, bool RowMajorB, bool RowMajorC>
void cpugemm(float *A, float *B, float *C, int M, int N, int K, int batch)
{
    for (int i = 0; i < batch; i++)
    {
        for (int k = 0; k < K; k++)
        {
            for (int m = 0; m < M; m++)
            {
                for (int n = 0; n < N; n++)
                {
                    float a = RowMajorA ? A[i * M * K + m * K + k] : A[i * M * K + k * M + m];
                    float b = RowMajorB ? B[i * N * K + k * N + n] : B[i * N * K + n * K + k];
                    unsigned ci = RowMajorC ? (i * M * N + m * N + n) : (i * M * N + n * M + m);
                    C[ci] += (float)a * (float)b;
                }
            }
        }
    }
}

template <bool RowMajorA, bool RowMajorB, bool RowMajorC>
bool verify(float *h_A, float *h_B, float *h_C, int M, int N, int K, int batch, float eps)
{
    bool correct = true;
    int mem_size_C = M * N * sizeof(float) * batch;
    float *realC = (float *)malloc(mem_size_C);
    memset(realC, 0, mem_size_C);
    cpugemm<RowMajorA, RowMajorB, RowMajorC>(h_A, h_B, realC, M, N, K, batch);

#if PRINT_MATRIX
    printf("Ref:\n");
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M - 1; i++)
        {
            printf("%4.f, ", realC[j * M + i]);
        }
        printf("%4.f\n", realC[j * M + M - 1]);
    }

    printf("\nCompute:\n");
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M - 1; i++)
        {
            printf("%4.f, ", (float)h_C[j * M + i]);
        }
        printf("%4.f\n", (float)h_C[j * M + M - 1]);
    }
#endif

    for (int i = 0; i < static_cast<int>(M * N); i++)
    {
        float abs_err = fabs((float)h_C[i] - realC[i]);
        float abs_val = fabs((float)h_C[i]);
        float rel_err = abs_err / (abs_val * K);

        if (rel_err > eps || std::isnan(realC[i]) || std::isnan((float)h_C[i]))
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %lf\n", i, (float)h_C[i], realC[i], eps);
            correct = false;
            break;
        }
    }
    free(realC);
    return correct;
}

void RandInit(float *data, unsigned size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-6.0, 6.0);

    for (unsigned i = 0; i < size; ++i)
    {
        data[i] = dist(gen);
    }
}

template <bool RowMajorA, bool RowMajorB, bool RowMajorC>
cudaError_t TestCutlassSgemm(int batch, int M, int N, int K, const float *d_A, const float *d_B, float *d_C)
{
    using ElementInputA = float; // <- data type of elements in input matrix A
    using ElementInputB = float; // <- data type of elements in input matrix B
    using ElementOutput = float; // <- data type of elements in output matrix D
    using ElementAccumulator = float;

    using OperatorClass = cutlass::arch::OpClassSimt;

    using ArchTag = cutlass::arch::Sm86;

    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,                                    // <- data type of output matrix
        128 / cutlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
                                                          // memory access. For a byte, it's 16
                                                          // elements. This becomes the vector width of
                                                          // math instructions in the epilogue too
        ElementAccumulator,                               // <- data type of accumulator
        ElementAccumulator>;                              // <- data type for alpha/beta in linear combination function

    // This code section describes how threadblocks are scheduled on GPU
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

    // Number of pipelines you want to use
    constexpr int NumStages = 2;

    using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                    cutlass::layout::RowMajor, // Layout of A matrix
                                                    float,        // Data-type of B matrix
                                                    cutlass::layout::RowMajor, // Layout of B matrix
                                                    float,        // Data-type of C matrix
                                                    cutlass::layout::RowMajor
                                                    // ElementAccumulator,
                                                    // OperatorClass,
                                                    // ArchTag
                                                    // ThreadblockShape
                                                    // WarpShape,
                                                    // InstructionShape,
                                                    // EpilogueOutputOp,
                                                    // SwizzleThreadBlock,
                                                    // NumStages
                                                    >; // Layout of C matrix

    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    const float alpha = 1.f;
    const float beta = 0.f;

    CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
                                {d_A, lda},     // Tensor-ref for source matrix A
                                {d_B, ldb},     // Tensor-ref for source matrix B
                                {d_C, ldc},     // Tensor-ref for source matrix C
                                {d_C, ldc},     // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    //
    // Launch the CUTLASS GEMM kernel.
    //
    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess)
    {
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
template <bool RowMajorA, bool RowMajorB, bool RowMajorC>
int MatrixMultiply(int batch, unsigned M, unsigned N, unsigned K, bool verifyResult)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = M * K * batch;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = new float[size_A]{};

    unsigned int size_B = K * N * batch;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = new float[size_B]{};

    RandInit(h_A, size_A);
    RandInit(h_B, size_B);

    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Allocate host matrix C
    unsigned int size_C = M * N * batch;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float *h_C = new float[size_C]{};

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
    CUDA_CHECK((TestCutlassSgemm<RowMajorA, RowMajorB, RowMajorC>(batch, M, N, K, d_A, d_B, d_C)));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start));

    // Execute the kernel
    int nIter = 10;
    for (int j = 0; j < nIter; j++)
    {
        CUDA_CHECK((TestCutlassSgemm<RowMajorA, RowMajorB, RowMajorC>(batch, M, N, K, d_A, d_B, d_C)));
    }

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop));
    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecTotal = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecTotal, start, stop));

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    float flopsPerMatrixMul = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) * batch;
    float gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    float bandWidth = (float)(M * K + N * K + M * N) * 2 / (msecPerMatrixMul * 1000 * 1000);
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
        MAKE_GEMM_4D_TEST(1, 1024, 1024, 1024, true, true, true),
        MAKE_GEMM_4D_TEST(1, 2048, 2048, 2048, true, true, true),
    };

    for (int i = 0; i < test_datas.size(); i++)
    {
        auto [batch, M, N, K, RowMajorA, RowMajorB, RowMajorC] = test_datas[i];

        printf("batch:%d, M:%d, N:%d, K:%d RowMajorA:%d RowMajorB:%d RowMajorC:%d \n", batch, M, N, K, RowMajorA, RowMajorB, RowMajorC);
        matrix_result = MatrixMultiply<true, true, true>(batch, M, N, K, verifyResult);
    }

    exit(matrix_result);
}