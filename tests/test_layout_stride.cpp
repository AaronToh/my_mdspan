#include <my_mdspan/layout_stride.hpp>
#include <array>
#include <cassert>

int main() {
    // --- strides matching layout_right (row-major) ---
    // extents 3x4, strides {4, 1}: (i,j) -> i*4 + j
    {
        my::extents<int, 3, 4> e;
        my::layout_stride::mapping m(e, std::array<int, 2>{4, 1});

        assert(m.required_span_size() == 12);

        assert(m(0, 0) == 0);
        assert(m(0, 1) == 1);
        assert(m(1, 0) == 4);
        assert(m(2, 3) == 11);
    }

    // --- strides matching layout_left (col-major) ---
    // extents 3x4, strides {1, 3}: (i,j) -> i + j*3
    {
        my::extents<int, 3, 4> e;
        my::layout_stride::mapping m(e, std::array<int, 2>{1, 3});

        assert(m.required_span_size() == 12);

        assert(m(0, 0) == 0);
        assert(m(1, 0) == 1);
        assert(m(0, 1) == 3);
        assert(m(2, 3) == 11);
    }

    // --- non-contiguous: every other row ---
    // extents 3x4, strides {8, 1}: rows are 8 apart in memory
    // (i,j) -> i*8 + j
    {
        my::extents<int, 3, 4> e;
        my::layout_stride::mapping m(e, std::array<int, 2>{8, 1});

        // required_span_size = (3-1)*8 + (4-1)*1 + 1 = 20
        assert(m.required_span_size() == 20);

        assert(m(0, 0) == 0);
        assert(m(0, 3) == 3);
        assert(m(1, 0) == 8);
        assert(m(1, 3) == 11);
        assert(m(2, 0) == 16);
        assert(m(2, 3) == 19);
    }

    // --- 3D with custom strides ---
    // extents 2x3x4, strides {12, 4, 1}: standard row-major for 3D
    // (i,j,k) -> i*12 + j*4 + k
    {
        my::extents<int, 2, 3, 4> e;
        my::layout_stride::mapping m(e, std::array<int, 3>{12, 4, 1});

        assert(m.required_span_size() == 24);

        assert(m(0, 0, 0) == 0);
        assert(m(0, 0, 1) == 1);
        assert(m(0, 1, 0) == 4);
        assert(m(1, 0, 0) == 12);
        assert(m(1, 2, 3) == 23);
    }

    // --- 3D col-major strides ---
    // extents 2x3x4, strides {1, 2, 6}: (i,j,k) -> i + j*2 + k*6
    {
        my::extents<int, 2, 3, 4> e;
        my::layout_stride::mapping m(e, std::array<int, 3>{1, 2, 6});

        assert(m(0, 0, 0) == 0);
        assert(m(1, 0, 0) == 1);
        assert(m(0, 1, 0) == 2);
        assert(m(0, 0, 1) == 6);
        assert(m(1, 2, 3) == 23);
    }

    return 0;
}
