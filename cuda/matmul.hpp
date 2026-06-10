#pragma once
#include "tensor_ref.hpp"

namespace cuda_my {

void matmul(TensorRef<float, 2> a, TensorRef<float, 2> b, TensorRef<float, 2> c);

} // namespace cuda_my
