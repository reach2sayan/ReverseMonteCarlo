#pragma once
#include <fullrmc/core/Types.hpp>
#include <fullrmc/core/BoundaryConditions.hpp>
#include <memory>

namespace fullrmc {

// ---- Type-erased move generator interface ----
struct IMoveGenerator {
    virtual ~IMoveGenerator() = default;
    virtual void generate(coords_t& coords,
                          std::span<const index_t> indices) = 0;
    virtual std::shared_ptr<IMoveGenerator> clone() const = 0;
};

// CRTP helper: concrete generators inherit this to auto-implement clone()
// and the virtual interface without boilerplate.
template<typename Derived>
struct MoveGeneratorBase : IMoveGenerator {
    void generate(coords_t& coords,
                  std::span<const index_t> indices) override {
        static_cast<Derived*>(this)->generate_impl(coords, indices);
    }
    std::shared_ptr<IMoveGenerator> clone() const override {
        return std::make_shared<Derived>(static_cast<const Derived&>(*this));
    }
};

// Utility: compute centroid of the given atom indices.
inline vec3_t centroid(const coords_t& coords,
                        std::span<const index_t> indices) noexcept {
    vec3_t c = vec3_t::Zero();
    for (auto i : indices) c += coords.row(i).transpose();
    return c / static_cast<real_t>(indices.size());
}

} // namespace fullrmc
