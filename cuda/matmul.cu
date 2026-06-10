#include "matmul.hpp"

namespace cuda_my {

#define BLOCK_SIZE 16

// --- Naive kernel ---
// One thread computes one output element C[i,j].
//
// Things to figure out:
//   - How do you map a thread to an (i, j) output coordinate?
//     (hint: blockIdx, blockDim, threadIdx)
//   - How do you guard against threads that land outside the matrix?
//   - The inner loop over k is identical to your CPU matmul. Does layout matter here?

__global__ void matmul_kernel(TensorRef<float, 2> a, TensorRef<float, 2> b, TensorRef<float, 2> c) {
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    if (row >= a.shape[0] || col >= b.shape[1]) {
        return;
    }

    float acc = 0.0f;
    for (int i = 0; i < a.shape[1]; i++) {
        acc += a(row, i) * b(i, col);
    }
    c(row, col) = acc;
}

// --- Launcher ---
// Decide on a block size, compute the grid dimensions, launch the kernel.
// Think about: if M=5 and blockDim=4, how many blocks do you need in that dimension?

void matmul(TensorRef<float, 2> a, TensorRef<float, 2> b, TensorRef<float, 2> c) {
    int R = a.shape[0];
    int C = b.shape[1];
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridsPerBlock((R + threadsPerBlock.x - 1) / threadsPerBlock.x, (C + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matmul_kernel<<<gridsPerBlock, threadsPerBlock>>>(a, b, c);
}

} // namespace cuda_my
