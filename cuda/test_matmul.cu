#include "matmul.hpp"
#include <cassert>
#include <cstdio>
#include <cublas_v2.h>

// Host-side test harness.
//
// Workflow for every GPU test:
//   1. Allocate host arrays, fill with data.
//   2. Allocate device arrays with cudaMalloc.
//   3. Copy host → device with cudaMemcpy.
//   4. Build TensorRef structs pointing at device memory.
//   5. Call the launcher.
//   6. Copy device → host with cudaMemcpy.
//   7. Assert on host values.
//   8. Free device memory with cudaFree.
//
// Think about: what strides does a row-major 2D array have?
//              what about col-major? Does your TensorRef support both?

int main() {
    // TODO: 2x2 * 2x2 = 2x2, values you can verify by hand
    {
        float a[4] = {2, 1, 1, 4};
        float b[6] = {1, 0, 2, 1, 0, 2};
        float c[6];

        int shape_a[2] = {2, 2};
        int shape_b[2] = {2, 3};
        int shape_c[2] = {2, 3};
        int stride_a[2] = {2, 1};
        int stride_b[2] = {1, 2};
        int stride_c[2] = {3, 1};

        float* d_a;
        cudaMalloc(&d_a, sizeof(float) * 4);
        float* d_b;
        cudaMalloc(&d_b, sizeof(float) * 6);
        float* d_c;
        cudaMalloc(&d_c, sizeof(float) * 6);

        cudaMemcpy(d_a, a, sizeof(float) * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(float) * 6, cudaMemcpyHostToDevice);

        cuda_my::TensorRef<float, 2> ta(d_a, shape_a, stride_a);
        cuda_my::TensorRef<float, 2> tb(d_b, shape_b, stride_b);
        cuda_my::TensorRef<float, 2> tc(d_c, shape_c, stride_c);

        cuda_my::matmul(ta, tb, tc);

        cudaMemcpy(c, d_c, sizeof(float) * 6, cudaMemcpyDeviceToHost);

        assert(c[0] == 2);
        assert(c[1] == 5);
        assert(c[2] == 2);
        assert(c[3] == 1);
        assert(c[4] == 6);
        assert(c[5] == 8);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    // --- benchmark: 1024x1024 * 1024x1024 ---
    {
        int M = 1024, K = 1024, N = 1024;
        int shape_a[2] = {M, K}, stride_a[2] = {K, 1};
        int shape_b[2] = {K, N}, stride_b[2] = {N, 1};
        int shape_c[2] = {M, N}, stride_c[2] = {N, 1};

        float* d_a; cudaMalloc(&d_a, sizeof(float) * M * K);
        float* d_b; cudaMalloc(&d_b, sizeof(float) * K * N);
        float* d_c; cudaMalloc(&d_c, sizeof(float) * M * N);

        cuda_my::TensorRef<float, 2> ta(d_a, shape_a, stride_a);
        cuda_my::TensorRef<float, 2> tb(d_b, shape_b, stride_b);
        cuda_my::TensorRef<float, 2> tc(d_c, shape_c, stride_c);

        // warmup
        cuda_my::matmul(ta, tb, tc);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        cuda_my::matmul(ta, tb, tc);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        printf("naive 1024x1024: %.3f ms\n", ms);

        cudaEventRecord(start);
        cuda_my::tiled_matmul(ta, tb, tc);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&ms, start, stop);
        printf("tiled 1024x1024: %.3f ms\n", ms);

        float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle;
        cublasCreate(&handle);

        // warmup: cuBLAS JIT-compiles the kernel on first call
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&ms, start, stop);
        printf("cublas 1024x1024: %.3f ms\n", ms);

        cublasDestroy(handle);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    printf("all tests passed\n");
    return 0;
}
