#pragma once

#include "ActionHistory.h"
#include "FrameBuffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace gameagent {

enum class FighterFacing {
    right,
    left,
};

struct MockMoveDefinition {
    std::string name;
    std::vector<std::vector<int>> states;
};

struct MockAgentDecision {
    std::vector<int> commands;
    std::string move_name;
    std::size_t step_index = 0;
    std::size_t step_count = 0;
};

std::vector<MockMoveDefinition> make_sub_zero_mock_moves(FighterFacing facing);

class MockAgent {
public:
    explicit MockAgent(
        std::optional<std::uint32_t> random_seed = std::nullopt,
        FighterFacing facing = FighterFacing::right
    );

    MockAgentDecision infer(
        const std::vector<std::shared_ptr<const BufferedFrame>>& observation,
        const std::vector<AppliedActionRecord>& applied_action_history
    );
    void reset_sequence();

private:
    std::vector<MockMoveDefinition> moves_;
    std::mt19937 random_engine_;
    std::mutex mutex_;
    std::optional<std::size_t> active_move_;
    std::size_t active_step_ = 0;
};

}  // namespace gameagent
