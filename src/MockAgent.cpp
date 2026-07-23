#include "MockAgent.h"

#include <stdexcept>

namespace gameagent {
namespace {

constexpr int kYHighPunch = 0;
constexpr int kXHighKick = 1;
constexpr int kBLowPunch = 2;
constexpr int kALowKick = 3;
constexpr int kDown = 7;
constexpr int kLeft = 8;
constexpr int kRight = 9;

}  // namespace

std::vector<MockMoveDefinition> make_sub_zero_mock_moves(FighterFacing facing) {
    const int forward = facing == FighterFacing::right ? kRight : kLeft;
    const int back = facing == FighterFacing::right ? kLeft : kRight;

    return {
        {
            .name = "Congelar",
            .states = {{kDown}, {forward}, {kBLowPunch}, {}, {}},
        },
        {
            .name = "Clone de Gelo",
            .states = {{kDown}, {back}, {kBLowPunch}, {}, {}},
        },
        {
            .name = "Chuva de Gelo (Frente)",
            .states = {
                {kDown}, {forward}, {kBLowPunch}, {kYHighPunch}, {}, {}
            },
        },
        {
            .name = "Chuva de Gelo (Perto)",
            .states = {
                {kDown}, {back}, {forward}, {kYHighPunch}, {}, {}
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
        },
    };
}

MockAgent::MockAgent(
    std::optional<std::uint32_t> random_seed,
    FighterFacing facing
)
    : moves_(make_sub_zero_mock_moves(facing)),
      random_engine_(random_seed.value_or(std::random_device{}())) {}

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
        std::uniform_int_distribution<std::size_t> move_distribution(
            0, moves_.size() - 1U
        );
        active_move_ = move_distribution(random_engine_);
        active_step_ = 0;
    }

    const MockMoveDefinition& move = moves_[*active_move_];
    MockAgentDecision decision{
        .commands = move.states[active_step_],
        .move_name = move.name,
        .step_index = active_step_ + 1U,
        .step_count = move.states.size(),
    };

    ++active_step_;
    if (active_step_ == move.states.size()) {
        active_move_.reset();
        active_step_ = 0;
    }
    return decision;
}

void MockAgent::reset_sequence() {
    std::lock_guard lock(mutex_);
    active_move_.reset();
    active_step_ = 0;
}

}  // namespace gameagent
