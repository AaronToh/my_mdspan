#include <my_mdspan/ops.hpp>
#include <cassert>

int main() {
    // --- 1D add ---
    {
        int a[4] = {1, 2, 3, 4};
        int b[4] = {10, 20, 30, 40};
        int c[4] = {};

        my::mdspan<int, my::extents<int, 4>> ma(a, my::extents<int, 4>{});
        my::mdspan<int, my::extents<int, 4>> mb(b, my::extents<int, 4>{});
        my::mdspan<int, my::extents<int, 4>> mc(c, my::extents<int, 4>{});

        my::cpu::add(ma, mb, mc);

        assert(mc(0) == 11);
        assert(mc(1) == 22);
        assert(mc(2) == 33);
        assert(mc(3) == 44);
    }

    // --- 2D add, row-major ---
    {
        int a[6] = {1, 2, 3, 4, 5, 6};
        int b[6] = {10, 20, 30, 40, 50, 60};
        int c[6] = {};

        my::mdspan<int, my::extents<int, 2, 3>> ma(a, my::extents<int, 2, 3>{});
        my::mdspan<int, my::extents<int, 2, 3>> mb(b, my::extents<int, 2, 3>{});
        my::mdspan<int, my::extents<int, 2, 3>> mc(c, my::extents<int, 2, 3>{});

        my::cpu::add(ma, mb, mc);

        assert(mc(0, 0) == 11);
        assert(mc(0, 1) == 22);
        assert(mc(0, 2) == 33);
        assert(mc(1, 0) == 44);
        assert(mc(1, 1) == 55);
        assert(mc(1, 2) == 66);
    }

    {
        int a[6] = {1, 2, 3, 4, 5, 6};
        int b[6] = {10, 20, 30, 40, 50, 60};
        int c[6] = {};   

        my::mdspan<int, my::extents<int, 2, 3>> ma(a, my::extents<int, 2, 3>{});
        my::mdspan<int, my::extents<int, 2, 3>, my::layout_left> mb(b, my::extents<int, 2, 3>{});
        my::mdspan<int, my::extents<int, 2, 3>> mc(c, my::extents<int, 2, 3>{});

        my::cpu::add(ma, mb, mc);

        assert(mc(0, 0) == 11);
        assert(mc(0, 1) == 32);
        assert(mc(0, 2) == 53);
        assert(mc(1, 0) == 24);
        assert(mc(1, 1) == 45);
        assert(mc(1, 2) == 66);
    }

    // --- reduce axis 0 (sum along rows): 2x3 -> 3 ---
    // {{1,2,3},{4,5,6}} -> {5,7,9}
    {
        int data[6] = {1, 2, 3, 4, 5, 6};
        int out[3] = {};

        my::mdspan<int, my::extents<int, 2, 3>> in(data, my::extents<int, 2, 3>{});
        my::mdspan<int, my::extents<int, 3>> result(out, my::extents<int, 3>{});

        my::cpu::reduce(in, result, 0, 0, [](int a, int b){ return a + b; });

        assert(result(0) == 5);
        assert(result(1) == 7);
        assert(result(2) == 9);
    }

    // --- reduce axis 1 (sum along cols): 2x3 -> 2 ---
    // {{1,2,3},{4,5,6}} -> {6,15}
    {
        int data[6] = {1, 2, 3, 4, 5, 6};
        int out[2] = {};

        my::mdspan<int, my::extents<int, 2, 3>> in(data, my::extents<int, 2, 3>{});
        my::mdspan<int, my::extents<int, 2>> result(out, my::extents<int, 2>{});

        my::cpu::reduce(in, result, 1, 0, [](int a, int b){ return a + b; });

        assert(result(0) == 6);
        assert(result(1) == 15);
    }

    {
        int data[24] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
        int out[6];

        my::mdspan<int, my::extents<int, 2, 3, 4>> in(data, my::extents<int, 2, 3, 4>{});
        my::mdspan<int, my::extents<int, 2, 3>, my::layout_left> result(out, my::extents<int, 2, 3>{});

        my::cpu::reduce(in, result, 2, 1, [](int a, int b){ return a * b; });

        assert(result(0, 0) == 24);
        assert(result(0, 1) == 1680);
        assert(result(0, 2) == 11880);
        assert(result(1, 0) == 43680);
        assert(result(1, 1) == 116280);
        assert(result(1, 2) == 255024);
 
    }

    {
        int a[4] = {2, 1, 1, 4};
        int b[6] = {1, 0, 2, 1, 0, 2};
        int c[6];

        my::mdspan<int, my::extents<int, 2, 2>> ma(a, my::extents<int, 2, 2>{});
        my::mdspan<int, my::extents<int, 2, 3>, my::layout_left> mb(b, my::extents<int, 2, 3>{});
        my::mdspan<int, my::extents<int, 2, 3>> mc(c, my::extents<int, 2, 3>{});
        
        my::cpu::matmul(ma, mb, mc);

        assert(mc(0, 0) == 2);
        assert(mc(0, 1) == 5);
        assert(mc(0, 2) == 2);
        assert(mc(1, 0) == 1);
        assert(mc(1, 1) == 6);
        assert(mc(1, 2) == 8);
    }

    return 0;
}
