#pragma once
#include "extents.hpp"

namespace my {

struct layout_stride {
    template <class Extents>
    struct mapping {
        using extents_type = Extents;
        using index_type = typename Extents::index_type;
        using rank_type = typename Extents::rank_type;
        using strides_type = std::array<index_type, Extents::rank()>;
        using layout_type = layout_stride;

        constexpr mapping() noexcept = default;
        constexpr mapping(const Extents& e, const strides_type& s) noexcept : extents_(e), strides_(s) {}

        constexpr const extents_type& extents() const noexcept { return extents_; }
        constexpr const strides_type& strides() const noexcept { return strides_; }

        constexpr index_type required_span_size() const noexcept {
            constexpr rank_type rank = Extents::rank();
            index_type size = 1;
            for (rank_type i = 0; i < rank; i++) {
                size += (extents_.extent(i) - 1) * strides_[i];
            }
            return size;
        }

        constexpr index_type stride(rank_type r) const noexcept {
            return strides_[r];
        }
        
        template <class... Indices>
        constexpr index_type operator()(Indices... vals) const noexcept {
            constexpr rank_type rank = Extents::rank();
            static_assert(rank == sizeof...(vals));
            std::array<index_type, rank> indices = { static_cast<index_type> (vals)... };
            index_type flat_index = 0;
            for (rank_type i = 0; i < rank; i++) {
                flat_index += strides_[i] * indices[i];
            }
            return flat_index;
        }

    private:
        extents_type extents_;
        strides_type strides_;
    };
};

} // namespace my
