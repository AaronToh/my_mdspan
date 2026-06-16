#pragma once
#include <array>
#include <cstddef>
#include <type_traits>

namespace my {

inline constexpr std::size_t dynamic_extent = static_cast<std::size_t>(-1);

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
        return ((Extents == dynamic_extent) + ... + 0);
    }

    // Default constructor
    constexpr extents() noexcept = default;

    // Constructor for dynamic extents: extents<int, dynamic_extent, 4>(3)
    template <class... DynamicExtents>
    constexpr extents(DynamicExtents... vals) noexcept {
        static_assert(rank_dynamic() == sizeof...(vals));
        dynamic_vals = { static_cast<index_type>(vals)... };
    }

    constexpr index_type extent(rank_type i) const noexcept {
        if (static_exts[i] != dynamic_extent) return static_cast<index_type>(static_exts[i]);
        size_t dynamic_i = 0;
        for (size_t j = 0; j < i; j++) {
            if (static_exts[j] == dynamic_extent) dynamic_i++;
        }
        return dynamic_vals[dynamic_i];
    }
private:
    static constexpr std::array<std::size_t, rank()> static_exts = { Extents... };
    std::array<index_type, rank_dynamic()> dynamic_vals{};
};
} // namespace my
