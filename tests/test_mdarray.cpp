#include <my_mdspan/mdarray.hpp>
#include <my_mdspan/ops.hpp>
#include <cassert>

int main() {
    // 1D
    {
        my::mdarray<int, my::extents<int, 4>> a(my::extents<int, 4>{});
        a(0) = 1; a(1) = 2; a(2) = 3; a(3) = 4;
        assert(a(0) == 1);
        assert(a(3) == 4);
    }

    // 2D row-major
    {
        my::mdarray<int, my::extents<int, 2, 3>> a(my::extents<int, 2, 3>{});
        a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
        a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;
        assert(a(0, 2) == 3);
        assert(a(1, 1) == 5);
    }

    // 2D col-major
    {
        my::mdarray<int, my::extents<int, 2, 3>, my::layout_left> a(my::extents<int, 2, 3>{});
        a(0, 0) = 1; a(1, 0) = 2;
        a(0, 1) = 3; a(1, 1) = 4;
        a(0, 2) = 5; a(1, 2) = 6;
        assert(a(1, 0) == 2);
        assert(a(0, 2) == 5);
    }

    // view() returns a usable mdspan
    {
        my::mdarray<int, my::extents<int, 3>> a(my::extents<int, 3>{});
        a(0) = 10; a(1) = 20; a(2) = 30;
        auto v = a.view();
        assert(v(1) == 20);
        v(2) = 99;
        assert(a(2) == 99);
    }

    // pass to an op via view()
    {
        my::mdarray<int, my::extents<int, 4>> a(my::extents<int, 4>{});
        my::mdarray<int, my::extents<int, 4>> b(my::extents<int, 4>{});
        my::mdarray<int, my::extents<int, 4>> c(my::extents<int, 4>{});
        a(0)=1; a(1)=2; a(2)=3; a(3)=4;
        b(0)=10; b(1)=20; b(2)=30; b(3)=40;
        auto va = a.view(); auto vb = b.view(); auto vc = c.view();
        my::add(va, vb, vc);
        assert(c(0) == 11);
        assert(c(3) == 44);
    }

    return 0;
}
