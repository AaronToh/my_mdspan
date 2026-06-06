#pragma once
#include "extents.hpp"

namespace my {

// layout_left = column-major (Fortran order)
// First index is contiguous in memory.
//
// For a 3x4 mdspan, element (i,j) is at offset: i + j*3
//
// TODO: implement mapping — same interface as layout_right::mapping
// but the offset formula is reversed.

struct layout_left {
    template <class Extents>
    struct mapping {
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using rank_type    = typename Extents::rank_type;
        using layout_type  = layout_left;

        constexpr mapping() noexcept = default;
        constexpr mapping(const Extents& e) noexcept : extents_(e) {}

        constexpr const extents_type& extents() const noexcept { return extents_; }

        constexpr index_type required_span_size() const noexcept {
            rank_type rank = Extents::rank();
            index_type product = 1;
            for (rank_type i = 0; i < rank; i++) {
                product *= extents_.extent(i);
            }
            return product;
        }
        
        constexpr index_type stride(rank_type r) const noexcept {
            constexpr rank_type rank = Extents::rank();
            index_type s = 1;
            for (rank_type i = 0; i < rank; i++) {
                s *= extents_.extent(i);
            }
            return s;
        }

        // TODO: operator() with reversed stride order
        template <class... Indices>
        constexpr index_type operator()(Indices... vals) const noexcept {
            constexpr rank_type rank = Extents::rank();
            static_assert(rank == sizeof...(vals));
            std::array<index_type, rank> indices = { static_cast<index_type>(vals)... };
            index_type flat_index = 0;
            index_type product = 1;
            for (rank_type i = 0; i < rank; i++) {
                flat_index += product * indices[i];
                product *= extents_.extent(i);
            }
            return flat_index;
        }

    private:
        extents_type extents_;
    };
};

} // namespace my
