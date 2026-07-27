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
#include <string_view>
#include <vector>

namespace gameagent {

enum class FighterFacing {
    right,
    left,
};

enum class FighterCharacter {
    subzero,
    scorpion,
    noob_saibot,
    sindel,
};

// Complete UMK3 identity vocabulary used to condition a controlled
// character's policy on the current opponent. The four FighterCharacter
// values above remain the only profiles with scripted curricula.
enum class OpponentCharacter : std::uint8_t {
    unknown = 0,
    cyrax,
    ermac,
    jade,
    jax,
    kabal,
    kano,
    kitana,
    kung_lao,
    liu_kang,
    mileena,
    motaro,
    nightwolf,
    noob_saibot,
    rain,
    reptile,
    scorpion,
    sektor,
    shang_tsung,
    shao_kahn,
    sheeva,
    sindel,
    smoke,
    human_smoke,
    sonya,
    stryker,
    subzero,
    classic_subzero,
    count,
};

inline constexpr std::size_t kOpponentCharacterFeatureCount =
    static_cast<std::size_t>(OpponentCharacter::count);

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
std::optional<FighterCharacter> parse_fighter_character(
    std::string_view value
);
std::string_view fighter_character_slug(FighterCharacter character);
std::string_view fighter_character_name(FighterCharacter character);
std::vector<std::string> supported_fighter_characters();
std::optional<OpponentCharacter> parse_opponent_character(
    std::string_view value
);
std::string_view opponent_character_slug(OpponentCharacter character);
std::string_view opponent_character_name(OpponentCharacter character);
std::vector<std::string> supported_opponent_characters();

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
};

}  // namespace gameagent
