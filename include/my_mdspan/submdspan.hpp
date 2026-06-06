#pragma once
#include "my_mdspan/mdspan.hpp"

namespace my {

struct full_extent_t { explicit full_extent_t() = default; };
inline constexpr my::full_extent_t full_extent {};

template <class IndexType, std::size_t N, std::size_t... Is>
auto make_dynamic_extents(std::array<IndexType, N> arr, std::index_sequence<Is...>) {
    return my::extents<IndexType, ((void)Is, my::dynamic_extent)...>(arr[Is]...);
}

template <
    class ElementType,
    class Extents,
    class LayoutPolicy,
    class AccessorPolicy,
    class... Slices
>
auto submdspan(const mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& src, Slices... slices) {
    using rank_type = typename Extents::rank_type;
    using index_type = typename Extents::index_type;
    static_assert(Extents::rank() == sizeof...(slices));
    constexpr rank_type rank = ((!std::is_same_v<Slices, int>) + ... + 0);
    auto result_extents = std::array<index_type, rank>{};
    auto result_strides = std::array<index_type, rank>{};
    std::size_t result_offs = 0;
    std::size_t out = 0;
    std::size_t dim = 0;
    auto process_one = [&]<class Slice>(Slice slice) {
        if constexpr (std::is_same_v<Slice, int>) {
            result_offs += static_cast<std::size_t>(slice) * src.mapping().stride(dim);
        } else if constexpr (std::is_same_v<Slice, my::full_extent_t>) {
            result_extents[out] = src.extent(dim);
            result_strides[out++] = src.mapping().stride(dim);
        } else if constexpr (std::is_same_v<Slice, std::pair<index_type, index_type>>) {
            result_offs += slice.first * src.mapping().stride(dim);
            result_extents[out] = slice.second - slice.first;
            result_strides[out++] = src.mapping().stride(dim);
        }
        dim++;
    };
    (process_one(slices), ...);
    auto e = make_dynamic_extents(result_extents, std::make_index_sequence<rank>{});
    layout_stride::mapping m(e, result_strides);
    return mdspan<ElementType, decltype(e), layout_stride>(src.data_handle() + result_offs, m);
}
}
