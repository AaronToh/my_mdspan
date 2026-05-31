#include <my_mdspan/mdspan.hpp>
#include <cassert>

int main() {
    // --- basic 2D access ---
    {
        int data[12] = {0,1,2,3, 4,5,6,7, 8,9,10,11};
        my::mdspan<int, my::extents<int, 3, 4>> m(data, my::extents<int, 3, 4>{});

        assert(m.extent(0) == 3);
        assert(m.extent(1) == 4);

        assert(m(0, 0) == 0);
        assert(m(0, 1) == 1);
        assert(m(1, 0) == 4);
        assert(m(1, 2) == 6);
        assert(m(2, 3) == 11);
    }

    // --- dynamic extents ---
    {
        int data[12] = {0,1,2,3, 4,5,6,7, 8,9,10,11};
        auto e = my::extents<int, my::dynamic_extent, my::dynamic_extent>(3, 4);
        my::mdspan<int, decltype(e)> m(data, e);

        assert(m.extent(0) == 3);
        assert(m.extent(1) == 4);
        assert(m(1, 2) == 6);
    }

    // --- size() ---
    {
        int data[12] = {};
        my::mdspan<int, my::extents<int, 3, 4>> m(data, my::extents<int, 3, 4>{});
        assert(m.size() == 12);
    }

    // --- layout_left (col-major) ---
    {
        int data[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
        my::mdspan<int, my::extents<int, 3, 4>, my::layout_left> m(data, my::extents<int, 3, 4>{});

        // col-major: (i,j) -> i + j*3
        assert(m(0, 0) == 0);
        assert(m(1, 0) == 1);
        assert(m(0, 1) == 3);
    }

    // --- data_handle() returns original pointer ---
    {
        int data[6] = {};
        my::mdspan<int, my::extents<int, 2, 3>> m(data, my::extents<int, 2, 3>{});
        assert(m.data_handle() == data);
    }

    return 0;
}
