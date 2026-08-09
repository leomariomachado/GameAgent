#pragma once

#include "CharacterTypes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace gameagent {

inline constexpr std::uint32_t kServerProtocolVersion = 2;
inline constexpr std::uint32_t kServerArchitectureVersion = 21;
inline constexpr std::size_t kMaximumEncounterOpponents = 4;

enum class EncounterType : std::uint8_t {
    normal = 0,
    endurance = 1,
    boss = 2,
};

enum class EncounterEvent : std::uint8_t {
    none = 0,
    stage_started = 1,
    stage_completed = 2,
    encounter_won = 3,
    encounter_lost = 4,
};

enum class AgentMode : std::uint8_t {
    training = 0,
    evaluation = 1,
};

enum class ExplorationMode : std::uint8_t {
    automatic = 0,
    manual_fixed = 1,
    automatic_from_value = 2,
};

enum class ResetReason : std::uint8_t {
    none = 0,
    player_defeated = 1,
    encounter_completed = 2,
    invalid_game_state = 3,
    stream_stalled = 4,
    stuck = 5,
    manual = 6,
};

enum class ResetState : std::uint8_t {
    idle = 0,
    requested = 1,
    command_sent = 2,
    client_executing = 3,
    waiting_stable_state = 4,
    completed = 5,
    failed = 6,
};

std::string_view encounter_type_slug(EncounterType value);
std::string_view encounter_event_slug(EncounterEvent value);
std::string_view agent_mode_slug(AgentMode value);
std::string_view exploration_mode_slug(ExplorationMode value);
std::string_view reset_reason_slug(ResetReason value);
std::string_view reset_state_slug(ResetState value);

std::optional<EncounterType> parse_encounter_type(std::string_view value);
std::optional<EncounterEvent> parse_encounter_event(std::string_view value);
std::optional<AgentMode> parse_agent_mode(std::string_view value);
std::optional<ExplorationMode> parse_exploration_mode(std::string_view value);
std::optional<ResetReason> parse_reset_reason(std::string_view value);

struct MessageMetadata {
    std::uint64_t request_id = 0;
    std::int64_t timestamp_ms = 0;
};

struct EncounterConfiguration {
    EncounterType type = EncounterType::normal;
    std::uint8_t opponent_count = 1;
    std::uint8_t current_stage = 0;
    std::array<OpponentCharacter, kMaximumEncounterOpponents> opponents{};

    [[nodiscard]] OpponentCharacter current_opponent() const;
    [[nodiscard]] std::uint8_t remaining_opponents() const;
    void validate() const;
};

struct ConfigureSessionRequest {
    MessageMetadata metadata;
    std::string player_character;
    std::optional<std::string> player_side;
    std::optional<std::string> initial_facing;
    EncounterConfiguration encounter;
};

struct UpdateEncounterStageRequest {
    MessageMetadata metadata;
    std::uint64_t episode_id = 0;
    std::uint64_t stage_transition_id = 0;
    std::uint8_t current_stage = 0;
    OpponentCharacter current_opponent = OpponentCharacter::unknown;
};

struct EncounterEventRequest {
    MessageMetadata metadata;
    std::uint64_t episode_id = 0;
    std::uint64_t stage_transition_id = 0;
    EncounterEvent event = EncounterEvent::none;
    std::uint8_t encounter_stage = 0;
};

struct SetAgentModeRequest {
    MessageMetadata metadata;
    AgentMode mode = AgentMode::training;
    bool reset_evaluation_session = false;
    bool apply_on_next_episode = true;
};

struct SetExplorationRequest {
    MessageMetadata metadata;
    ExplorationMode mode = ExplorationMode::automatic;
    float value = 0.0f;
    bool reset_matchup_counter = false;
};

struct RequestReset {
    MessageMetadata metadata;
    std::uint64_t episode_id = 0;
    ResetReason reason = ResetReason::manual;
};

struct ResetAck {
    MessageMetadata metadata;
    std::uint64_t reset_id = 0;
    std::uint64_t episode_id = 0;
    bool command_executed = false;
    std::int64_t executed_timestamp_ms = 0;
};

struct ResetReady {
    MessageMetadata metadata;
    std::uint64_t reset_id = 0;
    bool stable_initial_state = false;
    std::uint64_t first_stable_frame_id = 0;
};

struct ResetFailed {
    MessageMetadata metadata;
    std::uint64_t reset_id = 0;
    std::string reason;
};

struct EncounterTransition {
    bool accepted = false;
    bool duplicate = false;
    bool terminal = false;
    bool won = false;
    bool lost = false;
    bool opponent_health_delta_ignored = false;
    float stage_bonus = 0.0f;
    OpponentCharacter previous_opponent = OpponentCharacter::unknown;
    OpponentCharacter current_opponent = OpponentCharacter::unknown;
    std::string message;
};

struct ModeChangeResult {
    bool accepted = false;
    bool pending = false;
    bool changed = false;
    AgentMode effective_mode = AgentMode::training;
    std::optional<AgentMode> pending_mode;
};

struct ExplorationChangeResult {
    bool accepted = false;
    ExplorationMode mode = ExplorationMode::automatic;
    float value = 0.0f;
    bool matchup_counter_reset = false;
};

struct ResetTransition {
    bool accepted = false;
    bool duplicate = false;
    bool episode_aborted = false;
    ResetState state = ResetState::idle;
    std::uint64_t reset_id = 0;
    std::uint64_t episode_id = 0;
    std::uint64_t new_episode_id = 0;
    std::string message;
};

struct ModeStatistics {
    std::uint64_t episodes = 0;
    std::uint64_t wins = 0;
    std::uint64_t losses = 0;
    std::uint64_t aborted = 0;
    std::uint64_t stage_completions = 0;
    double episode_return_sum = 0.0;
    double damage_dealt_sum = 0.0;
    double damage_received_sum = 0.0;
    double episode_duration_ms_sum = 0.0;
    float best_episode_return = 0.0f;
    float worst_episode_return = 0.0f;
};

struct EpisodeStatistics {
    std::uint64_t episode_id = 0;
    AgentMode mode = AgentMode::training;
    float episode_return = 0.0f;
    float reward_sum = 0.0f;
    float best_step_reward = 0.0f;
    float worst_step_reward = 0.0f;
    float damage_dealt = 0.0f;
    float damage_received = 0.0f;
    std::uint64_t decisions = 0;
    std::uint64_t exploratory_decisions = 0;
    std::uint64_t greedy_decisions = 0;
    double epsilon_sum = 0.0;
    std::int64_t started_timestamp_ms = 0;
    bool won = false;
    bool lost = false;
    bool aborted = false;
    std::uint8_t stages_completed = 0;
};

struct ResetStatistics {
    std::uint64_t resets_requested = 0;
    std::uint64_t resets_completed = 0;
    std::uint64_t resets_failed = 0;
    std::uint64_t reset_timeouts = 0;
    std::uint64_t manual_resets = 0;
    std::uint64_t invalid_state_resets = 0;
    float last_reset_latency_ms = 0.0f;
    float mean_reset_latency_ms = 0.0f;
};

// Deterministic server-side state machine. It deliberately contains no socket
// or learner code, which makes encounter/reset semantics independently
// testable and keeps the server as the single source of truth.
class ServerSessionV21 {
public:
    explicit ServerSessionV21(
        AgentMode initial_mode = AgentMode::training,
        float epsilon_minimum = 0.05f
    );

    void configure(const EncounterConfiguration& encounter);
    void begin_episode(std::uint64_t episode_id, std::int64_t timestamp_ms);

    EncounterTransition apply_stage_update(
        const UpdateEncounterStageRequest& request
    );
    EncounterTransition apply_encounter_event(
        const EncounterEventRequest& request
    );

    ModeChangeResult request_agent_mode(const SetAgentModeRequest& request);
    ModeChangeResult apply_pending_mode();
    ExplorationChangeResult set_exploration(
        const SetExplorationRequest& request
    );

    ResetTransition request_reset(
        ResetReason reason,
        std::uint64_t episode_id,
        std::uint64_t reset_id,
        std::int64_t timestamp_ms,
        bool terminal_committed
    );
    ResetTransition mark_reset_command_sent(std::uint64_t reset_id);
    ResetTransition acknowledge_reset(const ResetAck& acknowledgement);
    ResetTransition mark_reset_ready(const ResetReady& ready);
    ResetTransition mark_reset_failed(const ResetFailed& failure);
    ResetTransition check_reset_timeout(
        std::int64_t now_ms,
        std::int64_t acknowledgement_timeout_ms,
        std::int64_t ready_timeout_ms
    );

    [[nodiscard]] const EncounterConfiguration& encounter() const;
    [[nodiscard]] AgentMode mode() const;
    [[nodiscard]] std::optional<AgentMode> pending_mode() const;
    [[nodiscard]] ExplorationMode exploration_mode() const;
    [[nodiscard]] float exploration_value() const;
    [[nodiscard]] float epsilon_minimum() const;
    [[nodiscard]] bool episode_active() const;
    [[nodiscard]] std::uint64_t episode_id() const;
    [[nodiscard]] ResetState reset_state() const;
    [[nodiscard]] std::uint64_t active_reset_id() const;
    [[nodiscard]] ResetReason reset_reason() const;
    [[nodiscard]] bool decisions_blocked() const;
    [[nodiscard]] bool consume_health_delta_ignore();

private:
    EncounterConfiguration encounter_;
    AgentMode mode_ = AgentMode::training;
    std::optional<AgentMode> pending_mode_;
    ExplorationMode exploration_mode_ = ExplorationMode::automatic;
    float exploration_value_ = 0.0f;
    float epsilon_minimum_ = 0.05f;
    bool episode_active_ = false;
    std::uint64_t episode_id_ = 0;
    std::uint64_t last_stage_transition_id_ = 0;
    bool ignore_opponent_health_delta_ = false;
    ResetState reset_state_ = ResetState::idle;
    ResetReason reset_reason_ = ResetReason::none;
    std::uint64_t active_reset_id_ = 0;
    std::uint64_t reset_episode_id_ = 0;
    std::int64_t reset_requested_timestamp_ms_ = 0;
    std::int64_t reset_acknowledged_timestamp_ms_ = 0;
};

}  // namespace gameagent
