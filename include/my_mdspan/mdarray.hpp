#pragma once
#include "mdspan.hpp"

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
        , storage_(new ElementType[mapping_.required_span_size()])
        , size_(mapping_.required_span_size()) {}

    mdarray(const mdarray& other) 
        : mapping_(other.mapping_)
        , storage_(new ElementType[other.size_])
        , size_(other.size_) {
        for (size_t i = 0; i < size_; i++) {
            storage_[i] = other.storage_[i];
        }
    }

    mdarray(mdarray&& other)
        : mapping_(other.mapping_)
        , storage_(other.storage_)
        , size_(other.size_) {
        other.storage_ = nullptr;
    }

    ~mdarray() { delete[] storage_; }

    mdarray& operator=(const mdarray& other) {
        if (this != &other) {
            delete[] storage_; // should be new memory (size might be different)
            mapping_ = other.mapping_;
            storage_ = new ElementType[other.size_];
            size_ = other.size_;
            for (size_t i = 0; i < size_; i++) {
                storage_[i] = other.storage_[i];
            }
        }
        return *this;
    }

    // && is an rvalue reference, ensures that not refernced elsewhere in program
    mdarray& operator=(mdarray&& other) {
        if (this != &other) {
            delete[] storage_;
            mapping_ = other.mapping_;
            storage_ = other.storage_;
            size_ = other.size_;
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

private:
    mapping_type mapping_;
    ElementType* storage_;
    size_t size_;
};

} // namespace my
