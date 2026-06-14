#pragma once
#include "mdspan.hpp"
#include <cassert>

namespace my {

namespace detail {

template <class A, class B, class C, class IndexArray, std::size_t... Is>
void add_base(const A& a, const B& b, C& c, IndexArray& idx, std::index_sequence<Is...>) {
    c(idx[Is]...) = a(idx[Is]...) + b(idx[Is]...);
}

template <class A, class B, class C, class IndexArray>
void add_helper(const A& a, const B& b, C& c, IndexArray& idx, size_t depth) {
    if (depth == C::rank()) {
        add_base(a, b, c, idx, std::make_index_sequence<C::rank()>{});
        return;
    }
    for (size_t j = 0; j < c.extent(depth); j++) {
        idx[depth] = j;
        add_helper(a, b, c, idx, depth + 1);
    }
}

template <
    class In,
    class Out,
    class ElementType,
    class BinaryOp,
    class IndexArrayIn,
    class IndexArrayOut,
    std::size_t... InIs,
    std::size_t... OutIs
>
void reduce_base(
    const In& in,
    Out& out,
    size_t axis,
    ElementType init,
    BinaryOp op,
    IndexArrayIn& idx_in,
    IndexArrayOut& idx_out,
    std::index_sequence<InIs...>,
    std::index_sequence<OutIs...>
) {
    ElementType running = init;
    for (size_t i = 0; i < in.extent(axis); i++) {
        idx_in[axis] = i;
        running = op(running, in(idx_in[InIs]...));
    }
    out(idx_out[OutIs]...) = running;
}

template <
    class In,
    class Out,
    class ElementType,
    class BinaryOp,
    class IndexArrayIn,
    class IndexArrayOut
>
void reduce_helper(
    const In& in,
    Out& out,
    size_t axis,
    ElementType init,
    BinaryOp op,
    IndexArrayIn& idx_in,
    IndexArrayOut& idx_out,
    size_t depth_in,
    size_t depth_out
) {
    if (depth_in == In::rank()) {
        reduce_base(in, out, axis, init, op, idx_in, idx_out,
            std::make_index_sequence<In::rank()>{},
            std::make_index_sequence<Out::rank()>{});
        return;
    }
    if (depth_in == axis) {
        reduce_helper(in, out, axis, init, op, idx_in, idx_out, depth_in + 1, depth_out);
        return;
    }
    for (size_t j = 0; j < in.extent(depth_in); j++) {
        idx_in[depth_in] = j;
        idx_out[depth_out] = j;
        reduce_helper(in, out, axis, init, op, idx_in, idx_out, depth_in + 1, depth_out + 1);
    }
}

} // namespace detail

namespace cpu {

template <
    class ElementType,
    class Extents,
    class LayoutPolicyA,
    class LayoutPolicyB,
    class LayoutPolicyC,
    class AccessorPolicy
>
void add(
    const mdspan<ElementType, Extents, LayoutPolicyA, AccessorPolicy>& a,
    const mdspan<ElementType, Extents, LayoutPolicyB, AccessorPolicy>& b,
    mdspan<ElementType, Extents, LayoutPolicyC, AccessorPolicy>& c
) {
    using index_type = typename Extents::index_type;
    std::array<index_type, Extents::rank()> idx{};
    detail::add_helper(a, b, c, idx, 0);
}

template <
    class ElementType,
    class ExtentsIn,
    class ExtentsOut,
    class LayoutPolicyIn,
    class LayoutPolicyOut,
    class AccessorPolicy,
    class BinaryOp
>
void reduce(
    const mdspan<ElementType, ExtentsIn, LayoutPolicyIn, AccessorPolicy>& in,
    mdspan<ElementType, ExtentsOut, LayoutPolicyOut, AccessorPolicy>& out,
    size_t axis,
    ElementType init,
    BinaryOp op
) {
    static_assert(ExtentsIn::rank() == ExtentsOut::rank() + 1);
    using index_type = typename ExtentsIn::index_type;
    std::array<index_type, ExtentsIn::rank()> idx_in{};
    std::array<index_type, ExtentsOut::rank()> idx_out{};
    detail::reduce_helper(in, out, axis, init, op, idx_in, idx_out, 0, 0);
}

template<
    class ElementType,
    class ExtentsA,
    class ExtentsB,
    class ExtentsC,
    class LayoutPolicyA,
    class LayoutPolicyB,
    class LayoutPolicyC,
    class AccessorPolicy
>
void matmul(
    const mdspan<ElementType, ExtentsA, LayoutPolicyA, AccessorPolicy>& a,
    const mdspan<ElementType, ExtentsB, LayoutPolicyB, AccessorPolicy>& b,
    mdspan<ElementType, ExtentsC, LayoutPolicyC, AccessorPolicy>& c
) {
    static_assert(ExtentsA::rank() == 2 && ExtentsB::rank() == 2);
    assert(a.extent(1) == b.extent(0));
    assert(c.extent(0) == a.extent(0) && c.extent(1) == b.extent(1));

    for (size_t i = 0; i < a.extent(0); i++) {
        for (size_t j = 0; j < b.extent(1); j++) {
            c(i, j) = 0;
            for (size_t k = 0; k < a.extent(1); k++) {
                c(i, j) += a(i, k) * b(k, j);
            }
        }
    }
}

} // namespace cpu

} // namespace my
