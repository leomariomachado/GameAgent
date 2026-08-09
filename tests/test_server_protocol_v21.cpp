#include "ServerProtocolV21.h"
#include "StreamProtocol.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <variant>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

gameagent::EncounterConfiguration endurance() {
    gameagent::EncounterConfiguration result;
    result.type = gameagent::EncounterType::endurance;
    result.opponent_count = 2;
    result.opponents[0] = gameagent::OpponentCharacter::jax;
    result.opponents[1] = gameagent::OpponentCharacter::nightwolf;
    return result;
}

}  // namespace

int main() {
    using namespace gameagent;
    try {
        const ClientTextMessage parsed = parse_client_text_message(R"JSON({
            "protocol_version":2,
            "request_id":7,
            "timestamp_ms":1780000000000,
            "type":"configure_session",
            "player_character":"noob_saibot",
            "player_side":"left",
            "initial_facing":"right",
            "encounter_type":"endurance",
            "opponent_count":2,
            "current_stage":0,
            "opponents":["jax","nightwolf"]
        })JSON");
        require(
            std::holds_alternative<ConfigureSessionRequest>(parsed),
            "configure_session V2.1 parse"
        );
        const auto& configuration =
            std::get<ConfigureSessionRequest>(parsed);
        require(
            configuration.encounter.current_opponent() ==
                OpponentCharacter::jax,
            "initial endurance opponent"
        );

        ServerSessionV21 session(AgentMode::training, 0.05f);
        session.configure(configuration.encounter);
        session.begin_episode(812, 1780000000001);

        const EncounterTransition first = session.apply_encounter_event({
            .metadata = {.request_id = 8, .timestamp_ms = 1780000000100},
            .episode_id = 812,
            .stage_transition_id = 44,
            .event = EncounterEvent::stage_completed,
            .encounter_stage = 0,
        });
        require(first.accepted && !first.terminal, "stage is nonterminal");
        require(std::abs(first.stage_bonus - 15.0f) < 1e-6f, "stage bonus");
        require(first.opponent_health_delta_ignored, "health refill ignored");
        require(
            session.encounter().current_opponent() ==
                OpponentCharacter::nightwolf,
            "next opponent active"
        );

        const EncounterTransition duplicate = session.apply_encounter_event({
            .metadata = {.request_id = 9, .timestamp_ms = 1780000000200},
            .episode_id = 812,
            .stage_transition_id = 44,
            .event = EncounterEvent::stage_completed,
            .encounter_stage = 0,
        });
        require(
            duplicate.accepted && duplicate.duplicate &&
                duplicate.stage_bonus == 0.0f,
            "stage bonus is idempotent"
        );

        const ModeChangeResult pending = session.request_agent_mode({
            .metadata = {.request_id = 10, .timestamp_ms = 1780000000300},
            .mode = AgentMode::evaluation,
            .reset_evaluation_session = true,
            .apply_on_next_episode = true,
        });
        require(pending.accepted && pending.pending, "mode change pending");
        require(session.mode() == AgentMode::training, "training remains active");

        const EncounterTransition terminal = session.apply_encounter_event({
            .metadata = {.request_id = 11, .timestamp_ms = 1780000000400},
            .episode_id = 812,
            .event = EncounterEvent::encounter_won,
            .encounter_stage = 1,
        });
        require(terminal.accepted && terminal.terminal && terminal.won,
                "final stage terminal");

        const ResetTransition requested = session.request_reset(
            ResetReason::encounter_completed,
            812,
            94,
            1780000000500,
            true
        );
        require(requested.accepted && session.decisions_blocked(),
                "reset blocks decisions");

        const ResetTransition acknowledged = session.acknowledge_reset({
            .metadata = {.request_id = 12, .timestamp_ms = 1780000000600},
            .reset_id = 94,
            .episode_id = 812,
            .command_executed = true,
            .executed_timestamp_ms = 1780000000590,
        });
        require(
            acknowledged.accepted &&
                acknowledged.state == ResetState::waiting_stable_state &&
                session.decisions_blocked(),
            "reset ack still blocks decisions"
        );

        const ResetTransition ready = session.mark_reset_ready({
            .metadata = {.request_id = 13, .timestamp_ms = 1780000000800},
            .reset_id = 94,
            .stable_initial_state = true,
            .first_stable_frame_id = 900,
        });
        require(ready.accepted && ready.new_episode_id == 813,
                "reset_ready starts next episode");
        require(!session.decisions_blocked(), "reset completed");
        require(session.mode() == AgentMode::evaluation,
                "pending evaluation applied at episode boundary");

        std::cout << "[PASS] Game Agent server protocol V2.1 state machine passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
