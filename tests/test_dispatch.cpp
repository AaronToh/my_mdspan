#include <my_mdspan/dispatch.hpp>
#include <cassert>

int main() {
    // CPU matmul via dispatch
    {
        int a[4] = {2, 1, 1, 4};
        int b[6] = {1, 0, 2, 1, 0, 2};
        int c[6];

        my::mdspan<int, my::extents<int, 2, 2>> ma(a, my::extents<int, 2, 2>{}, my::Device::CPU);
        my::mdspan<int, my::extents<int, 2, 3>, my::layout_left> mb(b, my::extents<int, 2, 3>{}, my::Device::CPU);
        my::mdspan<int, my::extents<int, 2, 3>> mc(c, my::extents<int, 2, 3>{}, my::Device::CPU);

        my::matmul(ma, mb, mc);

        assert(mc(0, 0) == 2);
        assert(mc(0, 1) == 5);
        assert(mc(0, 2) == 2);
        assert(mc(1, 0) == 1);
        assert(mc(1, 1) == 6);
        assert(mc(1, 2) == 8);
    }

#ifdef __CUDACC__
    // CUDA matmul via dispatch
    {
        float ha[4] = {2, 1, 1, 4};
        float hb[6] = {1, 0, 2, 1, 0, 2};
        float hc[6] = {};

        float *da, *db, *dc;
        cudaMalloc(&da, 4 * sizeof(float));
        cudaMalloc(&db, 6 * sizeof(float));
        cudaMalloc(&dc, 6 * sizeof(float));
        cudaMemcpy(da, ha, 4 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(db, hb, 6 * sizeof(float), cudaMemcpyHostToDevice);

        my::mdspan<float, my::extents<int, 2, 2>> ma(da, my::extents<int, 2, 2>{}, my::Device::CUDA);
        my::mdspan<float, my::extents<int, 2, 3>, my::layout_left> mb(db, my::extents<int, 2, 3>{}, my::Device::CUDA);
        my::mdspan<float, my::extents<int, 2, 3>> mc(dc, my::extents<int, 2, 3>{}, my::Device::CUDA);

        my::matmul(ma, mb, mc);

        cudaMemcpy(hc, dc, 6 * sizeof(float), cudaMemcpyDeviceToHost);

        assert(hc[0] == 2);
        assert(hc[1] == 5);
        assert(hc[2] == 2);
        assert(hc[3] == 1);
        assert(hc[4] == 6);
        assert(hc[5] == 8);

        cudaFree(da); cudaFree(db); cudaFree(dc);
    }
#endif

    return 0;
}
