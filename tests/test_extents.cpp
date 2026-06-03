#include <my_mdspan/extents.hpp>
#include <cassert>
#include <type_traits>

int main() {
    constexpr auto D = my::dynamic_extent;

    // --- rank() ---
    static_assert(my::extents<int>::rank() == 0);
    static_assert(my::extents<int, 3>::rank() == 1);
    static_assert(my::extents<int, 3, 4>::rank() == 2);
    static_assert(my::extents<int, 3, 4, 5>::rank() == 3);

    // --- rank_dynamic() ---
    static_assert(my::extents<int, 3, 4>::rank_dynamic() == 0);
    static_assert(my::extents<int, D, 4>::rank_dynamic() == 1);
    static_assert(my::extents<int, D, D>::rank_dynamic() == 2);
    static_assert(my::extents<int, 3, D, 5>::rank_dynamic() == 1);

    // --- static extent() ---
    {
        my::extents<int, 3, 4> e;
        assert(e.extent(0) == 3);
        assert(e.extent(1) == 4);
    }

    // --- dynamic extent() ---
    {
        my::extents<int, D, 4> e(3);
        assert(e.extent(0) == 3);
        assert(e.extent(1) == 4);
    }

    // --- all dynamic ---
    {
        my::extents<int, D, D> e(3, 4);
        assert(e.extent(0) == 3);
        assert(e.extent(1) == 4);
    }

    // --- associated types ---
    {
        using E = my::extents<int, 3, 4>;
        static_assert(std::is_same_v<E::index_type, int>);
        static_assert(std::is_same_v<E::size_type,  unsigned int>);
        static_assert(std::is_same_v<E::rank_type,  std::size_t>);
    }

    // --- zero-size property: all-static extents should be empty ---
    // Uncomment once you've solved the conditional storage problem:
    static_assert(std::is_empty_v<my::extents<int, 3, 4>>);

    return 0;
}
