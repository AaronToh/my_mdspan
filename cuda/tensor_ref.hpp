#pragma once

// Device-side analogue of mdspan.

// Things to think about:
//   - Every method that runs on the GPU needs a qualifier. Which one?
//   - How do you construct one of these on the host and pass it to a kernel?
//     (hint: the struct itself is copied by value into the kernel — that's fine
//      as long as it contains no host-only types)

namespace cuda_my {

template <typename T, int Rank>
struct TensorRef {
    // TODO: members
    T* ptr;
    int shape[Rank];
    int strides[Rank];

    __host__ __device__
    TensorRef(T* ptr, int shape[Rank], int strides[Rank]) : ptr(ptr) {
        for (size_t i = 0; i < Rank; i++) {
            this->shape[i] = shape[i];
            this->strides[i] = strides[i];
        }
    } 

    template <class... Indices>
    __host__ __device__
    T& operator()(Indices... vals) {
        int indices[] = {static_cast<int>(vals)...};
        int offset = 0;
        for (int i = 0; i < Rank; i++) {
            offset += indices[i] * strides[i];
        }
        return ptr[offset];
    }
};

} // namespace cuda_my
