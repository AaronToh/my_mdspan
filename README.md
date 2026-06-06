# my_mdspan

A from-scratch C++ implementation of [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) — the C++23 multidimensional array view.

## What is mdspan?

`mdspan` is a non-owning view into a flat block of memory, interpreted as a multidimensional array. Think of it as the C++ equivalent of a NumPy array view or a PyTorch tensor — it holds a pointer and a description of shape and memory layout, but owns nothing.

```cpp
int data[12] = {0,1,2,3,4,5,6,7,8,9,10,11};

// View it as a 3x4 row-major matrix
my::mdspan<int, my::extents<int, 3, 4>> m(data, my::extents<int, 3, 4>{});

m(1, 2); // → 6  (row 1, col 2)

// Slice out row 1
auto row = my::submdspan(m, 1, my::full_extent);
row(0); // → 4
row(3); // → 7
```

## Building

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## References

- [P0009](https://wg21.link/p0009) — the mdspan proposal
- [kokkos/mdspan](https://github.com/kokkos/mdspan) — reference implementation
