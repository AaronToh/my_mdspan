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
            // TODO
            return 0;
        }

        // TODO: operator() with reversed stride order

    private:
        extents_type extents_;
    };
};

} // namespace my
