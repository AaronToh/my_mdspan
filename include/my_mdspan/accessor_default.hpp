#pragma once
#include <type_traits>

namespace my {

// Default accessor - just wraps pointer dereference.
// Not interesting to implement yourself; it's intentionally trivial.
// The real design work is in extents + layout.
template <class ElementType>
struct default_accessor {
    using element_type   = ElementType;
    using data_handle_type = ElementType*;
    using reference      = ElementType&;
    using offset_policy  = default_accessor;

    constexpr default_accessor() noexcept = default;

    // Allow conversion from accessor<T> to accessor<const T>
    template <class OtherElementType>
        requires std::is_convertible_v<OtherElementType(*)[], ElementType(*)[]>
    constexpr default_accessor(default_accessor<OtherElementType>) noexcept {}

    constexpr reference access(data_handle_type p, std::size_t i) const noexcept {
        return p[i];
    }

    constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept {
        return p + i;
    }
};

} // namespace my
