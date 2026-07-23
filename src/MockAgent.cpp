#include "MockAgent.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

namespace gameagent {

MockAgent::MockAgent(std::optional<std::uint32_t> random_seed)
    : random_engine_(random_seed.value_or(std::random_device{}())) {}

std::vector<int> MockAgent::infer(
    const std::vector<std::shared_ptr<const BufferedFrame>>& observation,
    const std::vector<AppliedActionRecord>& applied_action_history
) {
    if (observation.empty()) {
        throw std::invalid_argument("mock inference requires at least one frame");
    }
    static_cast<void>(applied_action_history);
    std::lock_guard lock(mutex_);
    std::array<int, 10> available{};
    std::iota(available.begin(), available.end(), 0);
    std::shuffle(available.begin(), available.end(), random_engine_);
    std::uniform_int_distribution<std::size_t> count_distribution(3, 10);
    const std::size_t count = count_distribution(random_engine_);
    return {available.begin(), available.begin() + static_cast<std::ptrdiff_t>(count)};
}

}  // namespace gameagent
