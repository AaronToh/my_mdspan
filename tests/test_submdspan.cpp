#include <my_mdspan/submdspan.hpp>
#include <cassert>
#include <iostream>

int main() {
    int data[12] = {0,1,2,3,4,5,6,7,8,9,10,11};
    int data3d[24] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23};

    // --- fix a row (int slice on dim 0) ---
    // src is 3x4 row-major, fix row 1 -> result is rank-1 of size 4
    // expected elements: 4,5,6,7
    {
        my::mdspan<int, my::extents<int, 3, 4>> m(data, my::extents<int, 3, 4>());
        auto subm = my::submdspan(m, 1, my::full_extent);
        assert(subm(0) == 4);
        assert(subm(1) == 5);
        assert(subm(2) == 6);
        assert(subm(3) == 7);
    }

    // --- fix a column (int slice on dim 1) ---
    // src is 3x4 row-major, fix col 2 -> result is rank-1 of size 3
    // expected elements: 2,6,10
    {
        my::mdspan<int, my::extents<int, 3, 4>> m(data, my::extents<int, 3, 4>());
        auto subm = my::submdspan(m, my::full_extent, 2);
        assert(subm(0) == 2);
        assert(subm(1) == 6);
        assert(subm(2) == 10);
    }

    // --- subrange on dim 1 (pair slice) ---
    // src is 3x4 row-major, take cols 1..3 -> result is 3x2
    // expected: row 0 = {1,2}, row 1 = {5,6}, row 2 = {9,10}
    {
        my::mdspan<int, my::extents<int, 3, 4>> m(data, my::extents<int, 3, 4>());
        auto subm = my::submdspan(m, my::full_extent, std::pair<int,int>{1, 3});
        assert(subm(0, 0) == 1);
        assert(subm(0, 1) == 2);
        assert(subm(1, 0) == 5);
        assert(subm(1, 1) == 6);
        assert(subm(2, 0) == 9);
        assert(subm(2, 1) == 10);
    }

    {
        my::mdspan<int, my::extents<int, 3, 4>, my::layout_left> m(data, my::extents<int, 3, 4>{});
        auto subm = my::submdspan(m, 1, std::pair<int,int>{1, 3});
        assert(subm(0) == 4);
        assert(subm(1) == 7);
    }

    // --- 3D ---
    {
        my::mdspan<int, my::extents<int, 2, 3, 4>> m(data3d, my::extents<int, 2, 3, 4>());
        auto subm = my::submdspan(m, my::full_extent, 1, std::pair<int,int>{1, 3});
        assert(subm(0, 0) == 5);
        assert(subm(0, 1) == 6);
        assert(subm(1, 0) == 17);
        assert(subm(1, 1) == 18);
    }

    return 0;
}
