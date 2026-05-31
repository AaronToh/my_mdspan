#pragma once
#include <cstddef>
#include <type_traits>

namespace my {

inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

// TODO: implement extents
//
// Requirements:
//   - index_type  = IndexType
//   - size_type   = make_unsigned_t<IndexType>
//   - rank_type   = size_t
//   - static constexpr rank_type rank()         — number of dimensions
//   - static constexpr rank_type rank_dynamic() — how many are dynamic_extent
//   - constexpr index_type extent(rank_type i)  — size of dimension i
//
// Key design challenge:
//   Static extents should not occupy any storage (is_empty when rank_dynamic==0).
//   Dynamic extents need to store their runtime value somewhere.
//   Figure out how to do this with a single class template.

template <class IndexType, std::size_t... Extents>
class extents {
public:
    using index_type = IndexType;
    using size_type  = std::make_unsigned_t<IndexType>;
    using rank_type  = std::size_t;

    static constexpr rank_type rank() noexcept {
        return sizeof...(Extents);
    }

    static constexpr rank_type rank_dynamic() noexcept {
        // TODO
        return 0;
    }

    // Default constructor
    constexpr extents() noexcept = default;

    // Constructor for dynamic extents: extents<int, dynamic_extent, 4>(3)
    // TODO: figure out how to accept only the dynamic values

    constexpr index_type extent(rank_type i) const noexcept {
        // TODO
        (void)i;
        return 0;
    }
};

} // namespace my
