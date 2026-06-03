#pragma once
#include "extents.hpp"
#include "layout_right.hpp"
#include "layout_left.hpp"
#include "accessor_default.hpp"

namespace my {

// mdspan: a non-owning view into a multidimensional array.
// It holds:
//   - a data handle (pointer)
//   - a mapping (layout policy applied to extents)
//   - an accessor (how to turn pointer+offset into a reference)
//
// TODO: implement operator() to access elements
// TODO: implement size(), empty(), and the extents()/mapping() accessors

template <
    class ElementType,
    class Extents,
    class LayoutPolicy   = layout_right,
    class AccessorPolicy = default_accessor<ElementType>
>
class mdspan {
public:
    using extents_type     = Extents;
    using layout_type      = LayoutPolicy;
    using accessor_type    = AccessorPolicy;
    using mapping_type     = typename LayoutPolicy::template mapping<Extents>;
    using element_type     = ElementType;
    using data_handle_type = typename AccessorPolicy::data_handle_type;
    using reference        = typename AccessorPolicy::reference;
    using index_type       = typename Extents::index_type;
    using rank_type        = typename Extents::rank_type;

    static constexpr rank_type rank() noexcept { return Extents::rank(); }

    // Construct from pointer + extents
    constexpr mdspan(data_handle_type p, const Extents& e)
        : ptr_(p), mapping_(e), accessor_() {}

    // Construct from pointer + mapping
    constexpr mdspan(data_handle_type p, const mapping_type& m)
        : ptr_(p), mapping_(m), accessor_() {}

    // Element access: m(i, j, ...)
    // TODO: implement using mapping_(indices...) to get offset,
    //       then accessor_.access(ptr_, offset)
    template <class... Indices>
    constexpr element_type& operator()(Indices... vals) const noexcept {
        index_type i = mapping_(vals...);
        return accessor_.access(ptr_, i);
    }

    constexpr const extents_type& extents() const noexcept { return mapping_.extents(); }
    constexpr const mapping_type& mapping() const noexcept { return mapping_; }
    constexpr data_handle_type    data_handle() const noexcept { return ptr_; }

    constexpr index_type extent(rank_type r) const noexcept { return extents().extent(r); }

    constexpr index_type size() const noexcept {
        return mapping_.required_span_size();
    }

private:
    data_handle_type ptr_;
    mapping_type     mapping_;
    accessor_type    accessor_;
};

} // namespace my
