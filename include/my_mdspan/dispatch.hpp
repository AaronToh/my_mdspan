#pragma once
#include "device.hpp"
#include "mdspan.hpp"
#include "ops.hpp"
#ifdef __CUDACC__
#include "../../cuda/matmul.hpp"
#endif

#include <cassert>

namespace my {

template <class ElementType, class ExtentsA, class ExtentsB, class ExtentsC, class LayoutPolicyA, class LayoutPolicyB, class LayoutPolicyC>
void matmul(
    mdspan<ElementType, ExtentsA, LayoutPolicyA>& a,
    mdspan<ElementType, ExtentsB, LayoutPolicyB>& b,
    mdspan<ElementType, ExtentsC, LayoutPolicyC>& c
) {
    assert(a.device() == b.device());
    assert(b.device() == c.device());
    Device d = a.device();

    if (d == Device::CUDA) {
#ifdef __CUDACC__
        int shape_a[2] = {(int)a.extent(0), (int)a.extent(1)};
        int shape_b[2] = {(int)b.extent(0), (int)b.extent(1)};
        int shape_c[2] = {(int)c.extent(0), (int)c.extent(1)};
        int stride_a[2] = {(int)a.stride(0), (int)a.stride(1)};
        int stride_b[2] = {(int)b.stride(0), (int)b.stride(1)};
        int stride_c[2] = {(int)c.stride(0), (int)c.stride(1)};
        cuda_my::TensorRef<ElementType, 2> ta(a.data_handle(), shape_a, stride_a);
        cuda_my::TensorRef<ElementType, 2> tb(b.data_handle(), shape_b, stride_b);
        cuda_my::TensorRef<ElementType, 2> tc(c.data_handle(), shape_c, stride_c);
        cuda_my::matmul(ta, tb, tc);
#else
        throw std::runtime_error("CUDA not available");
#endif
    } else {
        my::cpu::matmul(a, b, c);
    }
}

} // namespace my
