#include "CombatStateTracker.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

gameagent::BufferedFrame make_frame(
    std::uint64_t frame_id,
    std::int64_t timestamp_ms,
    int player_x,
    int player_y,
    int opponent_x,
    int opponent_y
) {
    base::data::image::Image image(512, 448, 3);
    std::fill(image.bytes().begin(), image.bytes().end(), 12U);
    const auto rectangle = [&](
        int x,
        int y,
        std::array<std::uint8_t, 3> color
    ) {
        for (int row = std::max(70, y);
             row < std::min(410, y + 90);
             ++row) {
            for (int column = std::max(0, x);
                 column < std::min(512, x + 42);
                 ++column) {
                for (int channel = 0; channel < 3; ++channel) {
                    image.at(row, column, channel) =
                        color[static_cast<std::size_t>(channel)];
                }
            }
        }
    };
    rectangle(player_x, player_y, {210U, 225U, 245U});
    rectangle(opponent_x, opponent_y, {220U, 70U, 50U});
    return {
        .frame_id = frame_id,
        .client_timestamp_ms = timestamp_ms,
        .received_timestamp_ms = timestamp_ms + 2,
        .jpeg_size = 0,
        .image = std::move(image),
    };
}

gameagent::AppliedActionRecord action(
    std::initializer_list<int> commands
) {
    gameagent::AppliedActionRecord result;
    result.decision_id = 1;
    for (const int command : commands) {
        result.commands.push_back(command);
        result.command_state[static_cast<std::size_t>(command)] = 1.0f;
    }
    return result;
}

gameagent::FightControlSnapshot fight(
    bool episode_started = false
) {
    return {
        .episode_id = 1,
        .phase = gameagent::FightPhase::active_fight,
        .hud_valid = true,
        .action_allowed = true,
        .episode_started = episode_started,
        .player_health = 1.0f,
        .opponent_health = 1.0f,
        .hud_confidence = 1.0f,
    };
}

}  // namespace

int main() {
    try {
        gameagent::CombatStateTracker tracker({
            .player_side = gameagent::PlayerSide::left,
            .minimum_spatial_confidence = 0.02f,
            .position_smoothing = 1.0f,
            .idle_grace_ms = 100,
            .corner_grace_ms = 100,
            .combo_reward_scale = 0.05f,
            .defense_reward_per_second = 1.0f,
            .idle_penalty_per_second = 1.0f,
            .corner_penalty_per_second = 1.0f,
        });
        tracker.observe(
            make_frame(1, 1000, 120, 300, 360, 300),
            fight(true),
            action({})
        );
        auto state = tracker.observe(
            make_frame(2, 1100, 128, 300, 352, 300),
            fight(),
            action({4})
        );
        require(state.spatial_valid, "fighter motion was not tracked");
        require(
            state.player_x < state.opponent_x &&
                state.distance > 0.20f,
            "fighter positions or distance are invalid"
        );
        require(
            state.player_defending && !state.player_attacking,
            "confirmed block was not represented"
        );
        require(
            state.player_on_ground && state.opponent_on_ground,
            "grounded fighters were marked airborne"
        );

        auto damage = fight();
        damage.damage_dealt = 0.04f;
        state = tracker.observe(
            make_frame(3, 1200, 136, 300, 344, 300),
            damage,
            action({3})
        );
        require(
            state.player_attacking &&
                state.consecutive_hits == 1 &&
                std::abs(state.combo_reward) < 1.0e-6f,
            "first hit state is incorrect"
        );
        damage.damage_dealt = 0.03f;
        state = tracker.observe(
            make_frame(4, 1300, 144, 300, 336, 300),
            damage,
            action({3})
        );
        require(
            state.consecutive_hits == 2 &&
                std::abs(state.combo_reward - 0.05f) < 1.0e-5f,
            "consecutive-hit reward is incorrect"
        );

        auto received = fight();
        received.damage_taken = 0.03f;
        state = tracker.observe(
            make_frame(5, 1400, 152, 235, 328, 300),
            received,
            action({6})
        );
        require(
            !state.player_on_ground &&
                state.opponent_attacking &&
                state.opponent_action_confidence > 0.9f &&
                state.consecutive_hits == 0,
            "airborne/opponent attack state is incorrect"
        );
        const auto consumed = tracker.consume(1410);
        require(
            consumed.combo_reward >= 0.05f &&
                consumed.shaping_reward >= 0.05f,
            "pending combo shaping was not consumed"
        );
        require(
            std::abs(tracker.consume(1420).shaping_reward) <
                1.0e-6f,
            "combat shaping was consumed more than once"
        );

        state = tracker.observe(
            make_frame(6, 1500, 160, 300, 320, 300),
            fight(),
            action({4})
        );
        const auto defended = tracker.consume(1510);
        require(
            state.player_defending && state.opponent_attacking &&
                defended.defense_reward > 0.0f &&
                defended.shaping_reward >= defended.defense_reward,
            "successful contextual defense was not rewarded"
        );

        gameagent::CombatStateTracker penalties({
            .player_side = gameagent::PlayerSide::left,
            .minimum_spatial_confidence = 0.02f,
            .position_smoothing = 1.0f,
            .corner_boundary = 0.18f,
            .idle_grace_ms = 100,
            .corner_grace_ms = 100,
            .maximum_reward_delta_ms = 250,
            .combo_reward_scale = 0.0f,
            .idle_penalty_per_second = 1.0f,
            .corner_penalty_per_second = 1.0f,
        });
        penalties.observe(
            make_frame(10, 2000, 120, 300, 360, 300),
            fight(true),
            action({})
        );
        penalties.observe(
            make_frame(11, 2100, 80, 300, 350, 300),
            fight(),
            action({})
        );
        penalties.observe(
            make_frame(12, 2200, 45, 300, 340, 300),
            fight(),
            action({})
        );
        penalties.observe(
            make_frame(13, 2300, 25, 300, 330, 300),
            fight(),
            action({})
        );
        penalties.observe(
            make_frame(14, 2500, 25, 300, 330, 300),
            fight(),
            action({})
        );
        state = penalties.observe(
            make_frame(15, 2700, 25, 300, 330, 300),
            fight(),
            action({})
        );
        const auto penalty = penalties.consume(2710);
        require(
            state.player_x < 0.18f && state.corner_time_ms >= 200,
            "corner duration was not tracked"
        );
        require(
            penalty.idle_penalty > 0.0f &&
                penalty.corner_penalty > 0.0f &&
                penalty.shaping_reward < 0.0f,
            "idle/corner penalties were not accumulated"
        );

        std::cout
            << "[INFO] player_x=" << state.player_x
            << " opponent_x=" << state.opponent_x
            << " distance=" << state.distance
            << " spatial_confidence=" << state.spatial_confidence
            << " idle_penalty=" << penalty.idle_penalty
            << " corner_penalty=" << penalty.corner_penalty
            << '\n'
            << "[PASS] combat state and reward shaping test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
