#pragma once
#include "mdspan.hpp"
#include <vector>

namespace my {

// Owning counterpart to mdspan — holds its own data in a std::vector.
// Think about:
//   - What members does this need?
//   - How do you construct it from just an extents (allocate, no data given)?
//   - How do you expose a non-owning view (mdspan) over the owned data?

template <class ElementType, class Extents, class LayoutPolicy = layout_right>
class mdarray {
public:
    using extents_type = Extents;
    using mdspan_type = mdspan<ElementType, Extents, LayoutPolicy>;
    using mapping_type = typename LayoutPolicy::template mapping<Extents>;

    constexpr mdarray(const extents_type& e)
        : mapping_(mapping_type(e))
        , storage_(mapping_.required_span_size()) {}

    template <class... Indices>
    constexpr ElementType& operator()(Indices... vals) noexcept {
        return view()(vals...);
    }

    template <class... Indices>
    constexpr const ElementType& operator()(Indices... vals) const noexcept {
        return view()(vals...);
    }

    constexpr mdspan_type view() const noexcept {
        return mdspan_type(storage_.data(), mapping_);
    }

    constexpr mdspan_type view() noexcept {
        return mdspan_type(storage_.data(), mapping_);
    }

    constexpr const ElementType* data_handle() const noexcept {
        return storage_.data();
    }

private:
    mapping_type mapping_;
    std::vector<ElementType> storage_;
};

} // namespace my
