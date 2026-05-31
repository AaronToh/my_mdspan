#include <my_mdspan/layout_right.hpp>
#include <my_mdspan/layout_left.hpp>
#include <cassert>

int main() {
    // --- layout_right (row-major) ---
    // For a 3x4 array, element (i,j) -> i*4 + j
    {
        my::extents<int, 3, 4> e;
        my::layout_right::mapping m(e);

        assert(m.required_span_size() == 12);

        assert(m(0, 0) == 0);
        assert(m(0, 1) == 1);
        assert(m(1, 0) == 4);   // next row
        assert(m(1, 2) == 6);   // i*4 + j = 1*4 + 2
        assert(m(2, 3) == 11);  // last element
    }

    // --- layout_left (col-major) ---
    // For a 3x4 array, element (i,j) -> i + j*3
    {
        my::extents<int, 3, 4> e;
        my::layout_left::mapping m(e);

        assert(m.required_span_size() == 12);

        assert(m(0, 0) == 0);
        assert(m(1, 0) == 1);   // next row in same col
        assert(m(0, 1) == 3);   // next col
        assert(m(1, 2) == 7);   // i + j*3 = 1 + 2*3
        assert(m(2, 3) == 11);  // last element
    }

    // --- 3D layout_right ---
    // For shape (2,3,4), element (i,j,k) -> i*12 + j*4 + k
    {
        my::extents<int, 2, 3, 4> e;
        my::layout_right::mapping m(e);

        assert(m.required_span_size() == 24);
        assert(m(0, 0, 0) == 0);
        assert(m(0, 0, 1) == 1);
        assert(m(0, 1, 0) == 4);
        assert(m(1, 0, 0) == 12);
        assert(m(1, 2, 3) == 23); // last element
    }

    return 0;
}
