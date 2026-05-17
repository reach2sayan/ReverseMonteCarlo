#pragma once
#include <fullrmc/core/Types.hpp>
#include <memory>
#include <string>
#include <vector>

namespace fullrmc {

// Forward-declare the type-erased generator interface.
struct IMoveGenerator;

struct Group {
    std::string                       name;
    std::vector<index_t>              indices;   // atom indices belonging to this group
    std::shared_ptr<IMoveGenerator>   generator; // how this group moves
    bool                              refine = true; // participate in refinement?

    [[nodiscard]] std::size_t size()  const noexcept { return indices.size(); }
    [[nodiscard]] bool        empty() const noexcept { return indices.empty(); }
    [[nodiscard]] std::span<const index_t> span() const noexcept { return indices; }
};

} // namespace fullrmc
