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
    std::vector<std::string> phases;
    std::string action_source = "scripted_move";
    float curriculum_weight = 1.0f;
    bool conditional_freeze_follow_up = false;
    std::size_t freeze_confirmation_step = 0;
    std::vector<std::vector<int>> freeze_miss_states;
    std::vector<std::string> freeze_miss_phases;
};

struct MockAgentDecision {
    std::vector<int> commands;
    std::string move_name;
    std::size_t step_index = 0;
    std::size_t step_count = 0;
    std::string phase;
    std::string action_source;
    std::string branch;
    float freeze_visual_score = 0.0f;
    float freeze_visual_delta = 0.0f;
};

std::vector<MockMoveDefinition> make_sub_zero_mock_moves(FighterFacing facing);
std::vector<MockMoveDefinition> make_sub_zero_curriculum(FighterFacing facing);
float sub_zero_freeze_visual_score(
    const base::data::image::Image& image,
    FighterFacing facing
);

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
    void select_next_move(
        const std::vector<std::shared_ptr<const BufferedFrame>>& observation
    );

    FighterFacing facing_;
    std::vector<MockMoveDefinition> moves_;
    std::mt19937 random_engine_;
    std::mutex mutex_;
    std::optional<std::size_t> active_move_;
    std::vector<std::vector<int>> active_states_;
    std::vector<std::string> active_phases_;
    std::string active_branch_;
    float freeze_baseline_score_ = 0.0f;
    float freeze_latest_score_ = 0.0f;
    float freeze_delta_ = 0.0f;
    std::size_t active_step_ = 0;
};

}  // namespace gameagent
