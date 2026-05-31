# my_mdspan

A from-scratch implementation of `std::mdspan` (C++23) for learning purposes.
Reference: [kokkos/mdspan](https://github.com/kokkos/mdspan), [P0009 spec](https://wg21.link/p0009).

## Goal

Understand C++ template design by implementing a real STL component.
The focus is on `extents` (conditional static/dynamic storage), layout policies
(row-major vs col-major offset math), and how mdspan ties them together as a
non-owning view.

## What to implement

Work in this order. Each step has tests already written — make them pass.

### 1. `include/my_mdspan/extents.hpp`
The shape type. Holds the sizes of each dimension.

Key challenge: static extents (e.g. `extents<int, 3, 4>`) must occupy **zero bytes**
(`std::is_empty_v` must be true). Dynamic extents store their value at runtime.
The same class template must handle both.

Methods to implement:
- `rank()` — number of dimensions (compile-time)
- `rank_dynamic()` — how many dimensions are `dynamic_extent` (compile-time)
- `extent(i)` — size of dimension i (runtime)
- Constructor that accepts only the dynamic values: `extents<int, dynamic_extent, 4>(3)`

### 2. `include/my_mdspan/layout_right.hpp`
Row-major (C order) layout. Last index is contiguous in memory.

For shape `(3, 4)`, element `(i, j)` maps to flat offset `i*4 + j`.
For shape `(2, 3, 4)`, element `(i, j, k)` maps to `i*12 + j*4 + k`.

Implement inside `layout_right::mapping`:
- `required_span_size()` — product of all extents
- `operator()(i, j, ...)` — returns the flat offset

### 3. `include/my_mdspan/layout_left.hpp`
Column-major (Fortran order) layout. First index is contiguous in memory.

For shape `(3, 4)`, element `(i, j)` maps to flat offset `i + j*3`.
Same interface as `layout_right::mapping` but with reversed stride order.

### 4. `include/my_mdspan/mdspan.hpp`
Ties everything together. Non-owning view: pointer + mapping + accessor.

Implement:
- `operator()(i, j, ...)` — calls `mapping_(indices...)` to get offset,
  then `accessor_.access(ptr_, offset)` to get the reference
- `size()` — product of all extents (delegates to `mapping_.required_span_size()`)

## What is already done (do not modify)

- `include/my_mdspan/accessor_default.hpp` — trivial pointer accessor, not interesting to implement
- `CMakeLists.txt` — build system
- `tests/` — all tests are pre-written; just make them pass

## Building and running tests

```bash
cmake -B build -DCMAKE_CXX_STANDARD=23
cmake --build build
ctest --test-dir build --output-on-failure
```

Or run a single test:
```bash
./build/test_extents
./build/test_layout
./build/test_mdspan
```

## Stretch goal

Once all tests pass, uncomment this line in `tests/test_extents.cpp`:

```cpp
static_assert(std::is_empty_v<my::extents<int, 3, 4>>);
```

This will fail unless your extents type uses no storage when all dimensions are
static. Fixing it requires rethinking how you store values — look into
non-type template parameters and conditional member storage.

## Key concepts this exercises

- Variadic templates and parameter pack expansion (`sizeof...(Extents)`, fold expressions)
- Non-type template parameters
- Policy-based design (layout as a template parameter)
- `constexpr` and `static_assert`
- The difference between compile-time and runtime values in the type system

## Reference implementations to read (after you finish)

Read these only after your tests pass — not before.

- [kokkos/mdspan](https://github.com/kokkos/mdspan) — the reference impl that got standardized
- [LLVM libc++ tests](https://github.com/llvm/llvm-project/tree/main/libcxx/test/std/containers/views/mdspan) — the test suite your tests are modelled on
- PyTorch `TensorImpl` (`aten/src/ATen/TensorImpl.h`) — the runtime-only version of this same idea at production scale
