#pragma once
#include "extents.hpp"

namespace my {

// layout_right = row-major (C order)
// Last index is contiguous in memory.
//
// For a 3x4 mdspan, element (i,j) is at offset: i*4 + j
//
// TODO: implement the mapping struct inside layout_right.
// It must provide:
//   - required_span_size() -> total number of elements
//   - operator()(i, j, ...) -> flat offset into the data pointer

struct layout_right {
    template <class Extents>
    struct mapping {
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using rank_type    = typename Extents::rank_type;
        using layout_type  = layout_right;

        constexpr mapping() noexcept = default;
        constexpr mapping(const Extents& e) noexcept : extents_(e) {}

        constexpr const extents_type& extents() const noexcept { return extents_; }

        constexpr index_type required_span_size() const noexcept {
            // TODO: product of all extents
            return 0;
        }

        // operator() takes rank() indices and returns the flat offset
        // TODO

    private:
        extents_type extents_;
    };
};

} // namespace my
