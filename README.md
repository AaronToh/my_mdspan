# my_mdspan

A from-scratch C++ implementation of [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) — the C++23 multidimensional array view.

## Motivation

The project began as a learning exercise to implement `std::mdspan` from scratch. This turned out to be highly relevant to many of my areas of interest and led me to explore how it could be used for matrix operations like GEMM. I implemented a data-owning counterpart (`mdarray`), wrote CPU and CUDA matmul kernels, and built an API dispatch layer that routes operations to CPU or GPU based on where the data lives.

## Design

### Mdspan Template Parameters

Following [P0009](https://wg21.link/p0009), both rank and extents are template parameters. Knowing extents at compile time enables compiler optimisations and conserves stack space since the sizes are baked into the type. Many use cases like modelling physical 3D space allow us to know this in advance. Dynamic extents are also supported for cases where sizes are only known at runtime — `extents<int, dynamic_extent, 4>` stores one runtime value and bakes the `4` into the type.

### Submdspan

Implemented slicing via `submdspan` as described in [P0009](https://wg21.link/p0009), which returns a non-owning view into a subset of an mdspan. There are many use cases for this, for example passing a row of a 2D matrix as a subspan to a dot product function. Any rectangular subregion of a strided layout can be described with a `layout_stride` mapping by adjusting the base pointer and computing new strides from the slice parameters.

### Mdarray

`mdarray` is the data-owning counterpart to `mdspan`. Implemented the backing store as a raw pointer (rather than `std::vector`) to practice RAII and the Rule of Five: destructor, copy constructor, copy assignment, move constructor, and move assignment. CPU allocations use `new[]`/`delete[]`; CUDA allocations use `cudaMalloc`/`cudaFree`, guarded with `#ifdef __CUDACC__`.

### API Dispatch

Operations take `mdspan` rather than `mdarray`. By taking a non-owning view, the op doesn't care whether the data came from `mdarray`, a stack array, a `std::vector`, or a raw CUDA malloc. `mdarray::view()` exists to hand off a span when you want to call an op.

The device is stored on the span itself rather than passed as a parameter to the op. Passing device as a parameter is error-prone as the caller could pass the wrong device for the data. Storing it as an invariant of the span makes that class of bug impossible.

For GPU kernels, `mdspan` can't be used directly since it depends on STL and isn't `__device__`-compatible. Instead, `TensorRef` is a stripped-down GPU-side view with plain C arrays and `__host__ __device__` qualifiers. The dispatch layer builds a `TensorRef` from the mdspan's pointer and strides before launching the kernel.

### Matmul Kernels

Implemented two CUDA kernels. The naive kernel assigns one thread per output element and reads directly from global memory, giving poor memory bandwidth utilisation since each element of A and B is loaded many times.

The tiled kernel improves on this by loading 16×16 tiles of A and B into shared memory, so each element is only loaded from global memory once per tile.

Benchmarked on a 1024×1024 matrix multiply:

| Kernel | Time |
|--------|------|
| Naive  | 30.3 ms |
| Tiled  | 14.5 ms |
| cuBLAS | 0.6 ms |

The gap to cuBLAS comes down to tensor cores (dedicated matrix multiply hardware), larger tile sizes, and double buffering. Possible next steps: vectorised loads (`float4`) and `wmma` for tensor core access.

### Tests

Tests cover both the happy path and mixed-layout cases — row-major against column-major, custom strides, and 3D tensors. The dispatch test verifies that the same `my::matmul` call routes correctly to CPU or CUDA based on the span's device.

## Building

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## References

- [P0009](https://wg21.link/p0009) — the mdspan proposal
- [kokkos/mdspan](https://github.com/kokkos/mdspan) — reference implementation
