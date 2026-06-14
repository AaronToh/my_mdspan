#pragma once
#include "device.hpp"
#include "mdspan.hpp"
#include <stdexcept>

namespace my {

template <class ElementType, class Extents, class LayoutPolicy = layout_right>
class mdarray {
public:
    using extents_type = Extents;
    using mdspan_type = mdspan<ElementType, Extents, LayoutPolicy>;
    using mapping_type = typename LayoutPolicy::template mapping<Extents>;

    constexpr mdarray(const extents_type& e, Device d = Device::CPU)
        : mapping_(mapping_type(e))
        , storage_(nullptr)
        , size_(mapping_.required_span_size())
        , device_(d) {
        if (d == Device::CUDA) {
#ifdef __CUDACC__
            cudaMalloc(&storage_, size_ * sizeof(ElementType));
#else
            throw std::runtime_error("CUDA not available");
#endif
        } else {
            storage_ = new ElementType[size_];
        }
    }

    mdarray(const mdarray& other) 
        : mapping_(other.mapping_)
        , storage_(nullptr)
        , size_(other.size_)
        , device_(other.device_) {
        if (device_ == Device::CUDA) {
#ifdef __CUDACC__
            cudaMalloc(&storage_, size_ * sizeof(ElementType));
            cudaMemcpy(storage_, other.storage_, size_ * sizeof(ElementType), cudaMemcpyDeviceToDevice);
#else
            throw std::runtime_error("CUDA not available");
#endif
        } else {
            storage_ = new ElementType[size_];
            for (size_t i = 0; i < size_; i++) {
                storage_[i] = other.storage_[i];
            }
        }
    }

    mdarray(mdarray&& other)
        : mapping_(other.mapping_)
        , storage_(other.storage_)
        , size_(other.size_)
        , device_(other.device_) {
        other.storage_ = nullptr;
    }

    ~mdarray() {
        if (device_ == Device::CUDA) {
#ifdef __CUDACC__
            cudaFree(storage_);
#endif
        } else {
            delete[] storage_;
        }
    }

    mdarray& operator=(const mdarray& other) {
        assert(device_ == other.device_);
        if (this != &other) {
            mapping_ = other.mapping_;
            size_ = other.size_;
            if (device_ == Device::CUDA) {
#ifdef __CUDACC__
                cudaFree(storage_);
                cudaMalloc(&storage_, size_ * sizeof(ElementType));
                cudaMemcpy(storage_, other.storage_, size_ * sizeof(ElementType), cudaMemcpyDeviceToDevice);
#endif
            } else {
                delete[] storage_;
                storage_ = new ElementType[other.size_];
                for (size_t i = 0; i < size_; i++) {
                    storage_[i] = other.storage_[i];
                }
            }
            
        }
        return *this;
    }

    // && is an rvalue reference, ensures that not refernced elsewhere in program
    mdarray& operator=(mdarray&& other) {
        assert(device_ == other.device_);
        if (this != &other) {
            mapping_ = other.mapping_;
            size_ = other.size_;
            if (device_ == Device::CUDA) {
#ifdef __CUDACC__
                cudaFree(storage_);
#endif
            } else {
                delete[] storage_;
            }
            storage_ = other.storage_;
            other.storage_ = nullptr;
        }
        return *this;
    }

    template <class... Indices>
    constexpr ElementType& operator()(Indices... vals) noexcept {
        return view()(vals...);
    }

    template <class... Indices>
    constexpr const ElementType& operator()(Indices... vals) const noexcept {
        return view()(vals...);
    }

    constexpr mdspan_type view() const noexcept {
        return mdspan_type(storage_, mapping_);
    }

    constexpr mdspan_type view() noexcept {
        return mdspan_type(storage_, mapping_);
    }

    constexpr const ElementType* data_handle() const noexcept {
        return storage_;
    }

    constexpr Device device() const noexcept {
        return device_;
    }

private:
    Device device_;
    mapping_type mapping_;
    ElementType* storage_;
    size_t size_;
};

} // namespace my
