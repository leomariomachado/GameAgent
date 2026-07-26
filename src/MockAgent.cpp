#include "MockAgent.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gameagent {
namespace {

constexpr int kYHighPunch = 0;
constexpr int kXHighKick = 1;
constexpr int kBLowPunch = 2;
constexpr int kALowKick = 3;
constexpr int kBlock = 4;
constexpr int kRun = 5;
constexpr int kUp = 6;
constexpr int kDown = 7;
constexpr int kLeft = 8;
constexpr int kRight = 9;

}  // namespace

float sub_zero_freeze_visual_score(
    const base::data::image::Image& image,
    FighterFacing facing
) {
    if (image.empty() || image.channels() != 3) {
        return 0.0f;
    }
    const int x_begin =
        facing == FighterFacing::right ? image.width() / 2 : 0;
    const int x_end =
        facing == FighterFacing::right ? image.width() : image.width() / 2;
    const int y_begin = image.height() / 6;
    const int y_end = image.height() * 5 / 6;
    std::size_t selected = 0;
    std::size_t sampled = 0;
    for (int y = y_begin; y < y_end; y += 2) {
        for (int x = x_begin; x < x_end; x += 2) {
            const float red = image.at(y, x, 0);
            const float green = image.at(y, x, 1);
            const float blue = image.at(y, x, 2);
            const bool ice_colored =
                blue >= 100.0f &&
                blue >= red * 1.20f &&
                green >= red * 0.90f &&
                std::abs(blue - green) <= 105.0f;
            selected += ice_colored ? 1U : 0U;
            ++sampled;
        }
    }
    return sampled == 0
        ? 0.0f
        : static_cast<float>(selected) / static_cast<float>(sampled);
}

std::vector<MockMoveDefinition> make_sub_zero_mock_moves(FighterFacing facing) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    const int back = facing == FighterFacing::right ? kLeft : kRight;

    return {
        {
            .name = "Congelar",
            .states = {{kDown}, {forward}, {kBLowPunch}, {}, {}},
            .phases = {"special", "special", "special", "release", "observe"},
        },
        {
            .name = "Clone de Gelo",
            .states = {{kDown}, {back}, {kBLowPunch}, {}, {}},
            .phases = {"special", "special", "special", "release", "observe"},
        },
        {
            .name = "Chuva de Gelo (Frente)",
            .states = {
                {kDown}, {forward}, {kBLowPunch}, {kYHighPunch}, {}, {}
            },
            .phases = {
                "special", "special", "special", "special", "release", "observe"
            },
        },
        {
            .name = "Chuva de Gelo (Perto)",
            .states = {
                {kDown}, {back}, {forward}, {kYHighPunch}, {}, {}
            },
            .phases = {
                "special", "special", "special", "special", "release", "observe"
            },
        },
        {
            .name = "3 Golpes",
            .states = {
                {kYHighPunch},
                {},
                {kYHighPunch},
                {},
                {kDown, kBLowPunch},
                {},
                {},
            },
            .phases = {
                "combo", "release", "combo", "release",
                "combo", "release", "observe"
            },
        },
        {
            .name = "4 Golpes",
            .states = {
                {kXHighKick},
                {},
                {kXHighKick},
                {},
                {back, kXHighKick},
                {},
                {forward, kALowKick},
                {},
                {},
            },
            .phases = {
                "combo", "release", "combo", "release", "combo",
                "release", "combo", "release", "observe"
            },
        },
    };
}

std::vector<MockMoveDefinition> make_sub_zero_curriculum(
    FighterFacing facing
) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    const int back = facing == FighterFacing::right ? kLeft : kRight;
    auto moves = make_sub_zero_mock_moves(facing);
    for (auto& move : moves) {
        move.curriculum_weight = 5.0f;
    }

    moves.push_back({
        .name = "Congelar + Corrida + 3 Golpes",
        .states = {
            {kDown}, {forward}, {kBLowPunch}, {}, {}, {}, {},
            {kRun, forward}, {kRun, forward}, {forward}, {},
            {kYHighPunch}, {}, {kYHighPunch}, {},
            {kDown, kBLowPunch}, {}, {},
        },
        .phases = {
            "freeze", "freeze", "freeze", "release",
            "observe", "observe", "observe",
            "approach", "approach", "approach", "release",
            "combo", "release", "combo", "release",
            "combo", "release", "observe",
        },
        .action_source = "scripted_conditional",
        .curriculum_weight = 25.0f,
        .conditional_freeze_follow_up = true,
        .freeze_confirmation_step = 7,
        .freeze_miss_states = {
            {kDown}, {forward}, {kBLowPunch}, {}, {}, {}, {},
            {kBlock, back}, {kBlock}, {}, {},
        },
        .freeze_miss_phases = {
            "freeze", "freeze", "freeze", "release",
            "observe", "observe", "observe",
            "recover", "recover", "release", "observe",
        },
    });
    moves.push_back({
        .name = "Congelar + Corrida + 4 Golpes",
        .states = {
            {kDown}, {forward}, {kBLowPunch}, {}, {}, {}, {},
            {kRun, forward}, {kRun, forward}, {forward}, {},
            {kXHighKick}, {}, {kXHighKick}, {},
            {back, kXHighKick}, {}, {forward, kALowKick}, {}, {},
        },
        .phases = {
            "freeze", "freeze", "freeze", "release",
            "observe", "observe", "observe",
            "approach", "approach", "approach", "release",
            "combo", "release", "combo", "release", "combo",
            "release", "combo", "release", "observe",
        },
        .action_source = "scripted_conditional",
        .curriculum_weight = 25.0f,
        .conditional_freeze_follow_up = true,
        .freeze_confirmation_step = 7,
        .freeze_miss_states = {
            {kDown}, {forward}, {kBLowPunch}, {}, {}, {}, {},
            {kBlock, back}, {kBlock}, {}, {},
        },
        .freeze_miss_phases = {
            "freeze", "freeze", "freeze", "release",
            "observe", "observe", "observe",
            "recover", "recover", "release", "observe",
        },
    });

    const std::vector<MockMoveDefinition> combat_variations{
        {
            .name = "Pulo para Frente + Chute Alto",
            .states = {
                {kUp, forward},
                {kUp, forward},
                {forward},
                {forward, kXHighKick},
                {forward, kXHighKick},
                {},
                {},
            },
            .phases = {
                "jump", "jump", "airborne", "air_attack",
                "air_attack", "release", "observe",
            },
            .action_source = "scripted_variation",
            .curriculum_weight = 4.0f,
        },
        {
            .name = "Uppercut Agachado",
            .states = {
                {kDown},
                {kDown},
                {kDown, kYHighPunch},
                {kDown, kYHighPunch},
                {},
                {},
            },
            .phases = {
                "crouch", "crouch", "uppercut",
                "uppercut", "release", "observe",
            },
            .action_source = "scripted_variation",
            .curriculum_weight = 3.0f,
        },
        {
            .name = "Corrida + 3 Golpes",
            .states = {
                {kRun, forward},
                {kRun, forward},
                {kRun, forward},
                {forward},
                {},
                {kYHighPunch},
                {},
                {kYHighPunch},
                {},
                {kDown, kBLowPunch},
                {},
                {},
            },
            .phases = {
                "run", "run", "run", "approach", "release",
                "combo", "release", "combo", "release",
                "combo", "release", "observe",
            },
            .action_source = "scripted_variation",
            .curriculum_weight = 4.0f,
        },
        {
            .name = "Corrida + 4 Golpes",
            .states = {
                {kRun, forward},
                {kRun, forward},
                {kRun, forward},
                {forward},
                {},
                {kXHighKick},
                {},
                {kXHighKick},
                {},
                {back, kXHighKick},
                {},
                {forward, kALowKick},
                {},
                {},
            },
            .phases = {
                "run", "run", "run", "approach", "release",
                "combo", "release", "combo", "release",
                "combo", "release", "combo", "release", "observe",
            },
            .action_source = "scripted_variation",
            .curriculum_weight = 4.0f,
        },
    };
    moves.insert(
        moves.end(), combat_variations.begin(), combat_variations.end()
    );

    const std::vector<MockMoveDefinition> exploration{
        {
            .name = "Explorar: avançar e A",
            .states = {{forward}, {forward, kALowKick}, {}, {}},
            .phases = {"explore", "explore", "release", "observe"},
            .action_source = "random_exploration",
            .curriculum_weight = 1.0f,
        },
        {
            .name = "Explorar: agachar e B",
            .states = {{kDown}, {kDown, kBLowPunch}, {}, {}},
            .phases = {"explore", "explore", "release", "observe"},
            .action_source = "random_exploration",
            .curriculum_weight = 1.0f,
        },
        {
            .name = "Explorar: bloquear e recuar",
            .states = {{kBlock}, {kBlock, back}, {kBlock}, {}, {}},
            .phases = {
                "explore", "explore", "explore", "release", "observe"
            },
            .action_source = "random_exploration",
            .curriculum_weight = 1.0f,
        },
        {
            .name = "Explorar: corrida curta",
            .states = {{kRun, forward}, {kRun, forward}, {forward}, {}, {}},
            .phases = {
                "explore", "explore", "explore", "release", "observe"
            },
            .action_source = "random_exploration",
            .curriculum_weight = 1.0f,
        },
        {
            .name = "Explorar: X e liberar",
            .states = {{kXHighKick}, {}, {}, {}},
            .phases = {"explore", "release", "release", "observe"},
            .action_source = "random_exploration",
            .curriculum_weight = 1.0f,
        },
    };
    moves.insert(moves.end(), exploration.begin(), exploration.end());
    return moves;
}

MockAgent::MockAgent(
    std::optional<std::uint32_t> random_seed,
    FighterFacing facing
)
    : facing_(facing),
      moves_(make_sub_zero_curriculum(facing)),
      random_engine_(random_seed.value_or(std::random_device{}())) {}

void MockAgent::select_next_move(
    const std::vector<std::shared_ptr<const BufferedFrame>>& observation
) {
    std::vector<double> weights;
    weights.reserve(moves_.size());
    for (const auto& move : moves_) {
        weights.push_back(std::max(0.0f, move.curriculum_weight));
    }
    std::discrete_distribution<std::size_t> move_distribution(
        weights.begin(), weights.end()
    );
    active_move_ = move_distribution(random_engine_);
    const auto& move = moves_[*active_move_];
    active_states_ = move.states;
    active_phases_ = move.phases;
    active_branch_ = move.conditional_freeze_follow_up
        ? "freeze_pending"
        : "direct";
    active_step_ = 0;
    freeze_baseline_score_ =
        observation.empty() || !observation.back()
            ? 0.0f
            : sub_zero_freeze_visual_score(
                  observation.back()->image, facing_
              );
    freeze_latest_score_ = freeze_baseline_score_;
    freeze_delta_ = 0.0f;
}

MockAgentDecision MockAgent::infer(
    const std::vector<std::shared_ptr<const BufferedFrame>>& observation,
    const std::vector<AppliedActionRecord>& applied_action_history
) {
    if (observation.empty()) {
        throw std::invalid_argument("mock inference requires at least one frame");
    }
    static_cast<void>(applied_action_history);
    std::lock_guard lock(mutex_);

    if (!active_move_.has_value()) {
        select_next_move(observation);
    }

    const MockMoveDefinition& move = moves_[*active_move_];
    if (move.conditional_freeze_follow_up &&
        active_step_ == move.freeze_confirmation_step) {
        const bool has_image =
            !observation.empty() &&
            observation.back() &&
            !observation.back()->image.empty();
        freeze_latest_score_ = has_image
            ? sub_zero_freeze_visual_score(
                  observation.back()->image, facing_
              )
            : freeze_baseline_score_;
        freeze_delta_ =
            freeze_latest_score_ - freeze_baseline_score_;
        const bool confirmed =
            has_image &&
            freeze_latest_score_ >= 0.025f &&
            freeze_delta_ >= 0.010f;
        const bool clearly_missed =
            has_image &&
            freeze_delta_ < 0.004f;
        if (clearly_missed) {
            active_states_ = move.freeze_miss_states;
            active_phases_ = move.freeze_miss_phases;
            active_branch_ = "freeze_missed_recover";
        } else {
            active_branch_ = confirmed
                ? "freeze_confirmed_followup"
                : "freeze_ambiguous_followup";
        }
    }
    MockAgentDecision decision{
        .commands = active_states_[active_step_],
        .move_name = move.name,
        .step_index = active_step_ + 1U,
        .step_count = active_states_.size(),
        .phase =
            active_step_ < active_phases_.size()
                ? active_phases_[active_step_]
                : "unspecified",
        .action_source = move.action_source,
        .branch = active_branch_,
        .freeze_visual_score = freeze_latest_score_,
        .freeze_visual_delta = freeze_delta_,
    };

    ++active_step_;
    if (active_step_ == active_states_.size()) {
        active_move_.reset();
        active_states_.clear();
        active_phases_.clear();
        active_branch_.clear();
        active_step_ = 0;
    }
    return decision;
}

void MockAgent::reset_sequence() {
    std::lock_guard lock(mutex_);
    active_move_.reset();
    active_states_.clear();
    active_phases_.clear();
    active_branch_.clear();
    freeze_baseline_score_ = 0.0f;
    freeze_latest_score_ = 0.0f;
    freeze_delta_ = 0.0f;
    active_step_ = 0;
}

}  // namespace gameagent
