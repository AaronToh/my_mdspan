# CLAUDE.md — Developer Guide

## Collaboration style

The user wants to work through implementation decisions independently. Don't suggest solutions upfront — let them reason through the approach first. Guide when they are stuck or heading in the wrong direction (e.g. wrong mental model, non-compiling approach). Answer questions directly and concisely.

## Project

A from-scratch implementation of `std::mdspan` (C++23) for learning C++ template design.
Reference: [kokkos/mdspan](https://github.com/kokkos/mdspan), [P0009 spec](https://wg21.link/p0009).

## Building

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Single target:
```bash
cmake --build build --target test_extents
./build/test_extents
```

Only re-run `cmake -S . -B build` when `CMakeLists.txt` changes. For header-only changes, just rebuild.

## File structure

```
include/my_mdspan/
  extents.hpp          # shape type — static and dynamic dimension sizes
  layout_right.hpp     # row-major (C order) layout policy
  layout_left.hpp      # col-major (Fortran order) layout policy
  layout_stride.hpp    # arbitrary stride layout policy
  accessor_default.hpp # trivial pointer accessor (do not modify)
  mdspan.hpp           # ties everything together
  submdspan.hpp        # slicing: returns a sub-view of an mdspan

tests/
  test_extents.cpp
  test_layout.cpp
  test_layout_stride.cpp
  test_mdspan.cpp
  test_submdspan.cpp
```

## Architecture

Three layers compose to form an mdspan:

```
mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>
  │
  ├── ptr_       — raw pointer to data (non-owning)
  ├── mapping_   — LayoutPolicy::mapping<Extents>  (stores extents + computes offsets)
  └── accessor_  — default_accessor<ElementType>   (ptr[offset] → reference)
```

- **`extents`** — describes shape only. Static extents baked into the type; dynamic extents stored at runtime.
- **`layout_*::mapping`** — takes extents, computes flat offset from multi-dimensional indices. The layout struct itself is a stateless tag; the mapping is the stateful object.
- **`mdspan::operator()(i, j, ...)`** calls `mapping_(i, j, ...)` → offset, then `accessor_.access(ptr_, offset)` → reference.
