#pragma once

#include "ActionHistory.h"
#include "CharacterTypes.h"
#include "FrameBuffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace gameagent {

struct MockMoveDefinition {
    std::string name;
    std::vector<std::vector<int>> states;
    std::vector<std::string> phases;
    std::string action_source = "scripted_move";
    float curriculum_weight = 1.0f;
};

struct MockAgentDecision {
    std::vector<int> commands;
    std::string move_name;
    std::size_t step_index = 0;
    std::size_t step_count = 0;
    std::string phase;
    std::string action_source;
};

std::vector<MockMoveDefinition> make_sub_zero_mock_moves(FighterFacing facing);
std::vector<MockMoveDefinition> make_sub_zero_curriculum(FighterFacing facing);
std::vector<MockMoveDefinition> make_character_mock_moves(
    FighterCharacter character,
    FighterFacing facing
);
std::vector<MockMoveDefinition> make_character_curriculum(
    FighterCharacter character,
    FighterFacing facing
);
class MockAgent {
public:
    explicit MockAgent(
        std::optional<std::uint32_t> random_seed = std::nullopt,
        FighterFacing facing = FighterFacing::right,
        FighterCharacter character = FighterCharacter::subzero
    );

    MockAgentDecision infer(
        const std::vector<std::shared_ptr<const BufferedFrame>>& observation,
        const std::vector<AppliedActionRecord>& applied_action_history
    );
    void reset_sequence();

private:
    void select_next_move();

    std::vector<MockMoveDefinition> moves_;
    std::mt19937 random_engine_;
    std::mutex mutex_;
    std::optional<std::size_t> active_move_;
    std::vector<std::vector<int>> active_states_;
    std::vector<std::string> active_phases_;
    std::size_t active_step_ = 0;
    std::size_t disable_cloud_cooldown_moves_remaining_ = 0;
    std::size_t shared_close_move_cooldown_moves_remaining_ = 0;
};

}  // namespace gameagent
