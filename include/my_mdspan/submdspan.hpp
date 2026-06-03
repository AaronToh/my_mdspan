#pragma once
#include <my_mdspan/mdspan.hpp>

namespace my {

template <class IndexType, std::size_t... Is>
auto make_dynamic_extents_type(std::index_sequence<Is...>) {
    return my::extents<IndexType, ((void)Is, my::dynamic_extent)...>{};
}

// Todo: implement own full_extent
template <class IndexType, class DataHandleType, class... Slices>
auto make_dynamic_extents(DataHandleType src, Slices... slices) {
    auto tup = std::array<IndexType, sizeof...(Slices)>{};
    size_t out = 0;
    size_t dim = 0;
    auto process = [&]<class Slice>(Slice slice) {
        if constexpr (std;:is_same_v<Slice, std::int>) {
            
        } else if constexpr (std::is_same_v<Slice, std::full_extent>) {
            tup[out++] = src.extent(dim);
        } else if constexpr (std::is_same_v<Slice, std::pair<IndexType, IndexType>>) {
            tup[out++] = slice.second - slice.first;
        }
        dim++;
    };
    (process(slices), ...);
}

template <
    class ElementType,
    class Extents,
    class LayoutPolicy,
    class AccessorPolicy,
    class... SliceSpecs
>
auto submdspan(const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& src, SliceSpecs... slices) {
    static_assert(Extents::rank() == sizeof...(slices));
    constexpr Extents::rank_type rank = ((!std::is_same_v<SliceSpecs, int>) + ... + 0);

};
}
