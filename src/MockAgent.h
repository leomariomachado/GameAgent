#pragma once

#include "FrameBuffer.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <vector>

namespace gameagent {

class MockAgent {
public:
    explicit MockAgent(std::optional<std::uint32_t> random_seed = std::nullopt);

    std::vector<int> infer(
        const std::vector<std::shared_ptr<const BufferedFrame>>& observation
    );

private:
    std::mt19937 random_engine_;
    std::mutex mutex_;
};

}  // namespace gameagent
