#include "ServerProtocolV21.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace gameagent {
namespace {

template <typename Enum>
std::optional<Enum> parse_slug(
    std::string_view value,
    std::initializer_list<std::pair<std::string_view, Enum>> entries
) {
    for (const auto& [slug, result] : entries) {
        if (value == slug) {
            return result;
        }
    }
    return std::nullopt;
}

bool reset_requires_committed_terminal(ResetReason reason) {
    return reason == ResetReason::player_defeated ||
        reason == ResetReason::encounter_completed;
}

}  // namespace

std::string_view encounter_type_slug(EncounterType value) {
    switch (value) {
        case EncounterType::normal: return "normal";
        case EncounterType::endurance: return "endurance";
        case EncounterType::boss: return "boss";
    }
    return "unknown";
}

std::string_view encounter_event_slug(EncounterEvent value) {
    switch (value) {
        case EncounterEvent::none: return "none";
        case EncounterEvent::stage_started: return "stage_started";
        case EncounterEvent::stage_completed: return "stage_completed";
        case EncounterEvent::encounter_won: return "encounter_won";
        case EncounterEvent::encounter_lost: return "encounter_lost";
    }
    return "unknown";
}

std::string_view agent_mode_slug(AgentMode value) {
    return value == AgentMode::training ? "training" : "evaluation";
}

std::string_view exploration_mode_slug(ExplorationMode value) {
    switch (value) {
        case ExplorationMode::automatic: return "automatic";
        case ExplorationMode::manual_fixed: return "manual_fixed";
        case ExplorationMode::automatic_from_value:
            return "automatic_from_value";
    }
    return "unknown";
}

std::string_view reset_reason_slug(ResetReason value) {
    switch (value) {
        case ResetReason::none: return "none";
        case ResetReason::player_defeated: return "player_defeated";
        case ResetReason::encounter_completed: return "encounter_completed";
        case ResetReason::invalid_game_state: return "invalid_game_state";
        case ResetReason::stream_stalled: return "stream_stalled";
        case ResetReason::stuck: return "stuck";
        case ResetReason::manual: return "manual";
    }
    return "unknown";
}

std::string_view reset_state_slug(ResetState value) {
    switch (value) {
        case ResetState::idle: return "idle";
        case ResetState::requested: return "requested";
        case ResetState::command_sent: return "command_sent";
        case ResetState::client_executing: return "client_executing";
        case ResetState::waiting_stable_state: return "waiting_stable_state";
        case ResetState::completed: return "completed";
        case ResetState::failed: return "failed";
    }
    return "unknown";
}

std::optional<EncounterType> parse_encounter_type(std::string_view value) {
    return parse_slug<EncounterType>(value, {
        {"normal", EncounterType::normal},
        {"endurance", EncounterType::endurance},
        {"boss", EncounterType::boss},
    });
}

std::optional<EncounterEvent> parse_encounter_event(std::string_view value) {
    return parse_slug<EncounterEvent>(value, {
        {"none", EncounterEvent::none},
        {"stage_started", EncounterEvent::stage_started},
        {"stage_completed", EncounterEvent::stage_completed},
        {"encounter_won", EncounterEvent::encounter_won},
        {"encounter_lost", EncounterEvent::encounter_lost},
    });
}

std::optional<AgentMode> parse_agent_mode(std::string_view value) {
    return parse_slug<AgentMode>(value, {
        {"training", AgentMode::training},
        {"evaluation", AgentMode::evaluation},
    });
}

std::optional<ExplorationMode> parse_exploration_mode(
    std::string_view value
) {
    return parse_slug<ExplorationMode>(value, {
        {"automatic", ExplorationMode::automatic},
        {"manual_fixed", ExplorationMode::manual_fixed},
        {"automatic_from_value", ExplorationMode::automatic_from_value},
    });
}

std::optional<ResetReason> parse_reset_reason(std::string_view value) {
    return parse_slug<ResetReason>(value, {
        {"none", ResetReason::none},
        {"player_defeated", ResetReason::player_defeated},
        {"encounter_completed", ResetReason::encounter_completed},
        {"invalid_game_state", ResetReason::invalid_game_state},
        {"stream_stalled", ResetReason::stream_stalled},
        {"stuck", ResetReason::stuck},
        {"manual", ResetReason::manual},
    });
}

OpponentCharacter EncounterConfiguration::current_opponent() const {
    validate();
    return opponents[current_stage];
}

std::uint8_t EncounterConfiguration::remaining_opponents() const {
    validate();
    return static_cast<std::uint8_t>(opponent_count - current_stage - 1U);
}

void EncounterConfiguration::validate() const {
    const bool one_opponent =
        type == EncounterType::normal || type == EncounterType::boss;
    if ((one_opponent && opponent_count != 1U) ||
        (type == EncounterType::endurance && opponent_count < 2U) ||
        opponent_count > kMaximumEncounterOpponents ||
        current_stage >= opponent_count) {
        throw std::invalid_argument("invalid encounter opponent count/stage");
    }
    for (std::size_t index = 0; index < opponent_count; ++index) {
        if (opponents[index] == OpponentCharacter::unknown) {
            throw std::invalid_argument(
                "encounter opponents cannot contain unknown or holes"
            );
        }
    }
}

ServerSessionV21::ServerSessionV21(
    AgentMode initial_mode,
    float epsilon_minimum
) : mode_(initial_mode), epsilon_minimum_(epsilon_minimum) {
    if (!std::isfinite(epsilon_minimum_) || epsilon_minimum_ < 0.0f ||
        epsilon_minimum_ > 1.0f) {
        throw std::invalid_argument("epsilon minimum must be within 0..1");
    }
}

void ServerSessionV21::configure(
    const EncounterConfiguration& encounter
) {
    encounter.validate();
    encounter_ = encounter;
    last_stage_transition_id_ = 0;
    ignore_opponent_health_delta_ = false;
}

void ServerSessionV21::begin_episode(
    std::uint64_t episode_id,
    std::int64_t
) {
    if (episode_id == 0) {
        throw std::invalid_argument("episode_id must be positive");
    }
    apply_pending_mode();
    episode_id_ = episode_id;
    episode_active_ = true;
    reset_state_ = ResetState::idle;
    reset_reason_ = ResetReason::none;
    active_reset_id_ = 0;
    reset_episode_id_ = 0;
    encounter_.current_stage = 0;
    last_stage_transition_id_ = 0;
    ignore_opponent_health_delta_ = false;
}

EncounterTransition ServerSessionV21::apply_stage_update(
    const UpdateEncounterStageRequest& request
) {
    EncounterTransition result{
        .previous_opponent = encounter_.current_opponent(),
        .current_opponent = encounter_.current_opponent(),
    };
    if (!episode_active_ || request.episode_id != episode_id_) {
        result.message = "stage update does not match the active episode";
        return result;
    }
    if (request.stage_transition_id == 0) {
        result.message = "stage_transition_id must be positive";
        return result;
    }
    if (request.stage_transition_id <= last_stage_transition_id_) {
        result.accepted = true;
        result.duplicate = true;
        result.message = "stage transition already applied";
        return result;
    }
    if (encounter_.type != EncounterType::endurance ||
        request.current_stage != encounter_.current_stage + 1U ||
        request.current_stage >= encounter_.opponent_count ||
        request.current_opponent !=
            encounter_.opponents[request.current_stage]) {
        result.message = "invalid endurance stage transition";
        return result;
    }
    encounter_.current_stage = request.current_stage;
    last_stage_transition_id_ = request.stage_transition_id;
    ignore_opponent_health_delta_ = true;
    result.accepted = true;
    result.stage_bonus = 15.0f;
    result.opponent_health_delta_ignored = true;
    result.current_opponent = encounter_.current_opponent();
    return result;
}

EncounterTransition ServerSessionV21::apply_encounter_event(
    const EncounterEventRequest& request
) {
    EncounterTransition result{
        .previous_opponent = encounter_.current_opponent(),
        .current_opponent = encounter_.current_opponent(),
    };
    if (!episode_active_ || request.episode_id != episode_id_) {
        result.message = "encounter event does not match the active episode";
        return result;
    }
    if (request.event == EncounterEvent::stage_completed) {
        UpdateEncounterStageRequest update{
            .metadata = request.metadata,
            .episode_id = request.episode_id,
            .stage_transition_id = request.stage_transition_id,
            .current_stage = static_cast<std::uint8_t>(
                request.encounter_stage + 1U
            ),
            .current_opponent = request.encounter_stage + 1U <
                    encounter_.opponent_count
                ? encounter_.opponents[request.encounter_stage + 1U]
                : OpponentCharacter::unknown,
        };
        if (request.stage_transition_id > last_stage_transition_id_ &&
            request.encounter_stage != encounter_.current_stage) {
            result.message = "stage_completed does not match current stage";
            return result;
        }
        return apply_stage_update(update);
    }
    if (request.event == EncounterEvent::stage_started) {
        result.accepted = request.encounter_stage == encounter_.current_stage;
        result.message = result.accepted
            ? "stage start recorded"
            : "stage_started does not match current stage";
        return result;
    }
    if (request.event == EncounterEvent::encounter_won ||
        request.event == EncounterEvent::encounter_lost) {
        if (request.event == EncounterEvent::encounter_won &&
            encounter_.current_stage + 1U != encounter_.opponent_count) {
            result.message = "encounter cannot be won before its final stage";
            return result;
        }
        result.accepted = true;
        result.terminal = true;
        result.won = request.event == EncounterEvent::encounter_won;
        result.lost = request.event == EncounterEvent::encounter_lost;
        episode_active_ = false;
        return result;
    }
    result.message = "encounter event none is not actionable";
    return result;
}

ModeChangeResult ServerSessionV21::request_agent_mode(
    const SetAgentModeRequest& request
) {
    ModeChangeResult result{.accepted = true, .effective_mode = mode_};
    if (request.apply_on_next_episode && episode_active_) {
        pending_mode_ = request.mode;
        result.pending = true;
        result.pending_mode = pending_mode_;
        return result;
    }
    result.changed = mode_ != request.mode;
    mode_ = request.mode;
    pending_mode_.reset();
    result.effective_mode = mode_;
    return result;
}

ModeChangeResult ServerSessionV21::apply_pending_mode() {
    ModeChangeResult result{.accepted = true, .effective_mode = mode_};
    if (pending_mode_.has_value()) {
        result.changed = mode_ != *pending_mode_;
        mode_ = *pending_mode_;
        pending_mode_.reset();
        result.effective_mode = mode_;
    }
    return result;
}

ExplorationChangeResult ServerSessionV21::set_exploration(
    const SetExplorationRequest& request
) {
    if (!std::isfinite(request.value) || request.value < 0.0f ||
        request.value > 1.0f ||
        (request.mode == ExplorationMode::automatic_from_value &&
         request.value < epsilon_minimum_)) {
        return {};
    }
    exploration_mode_ = request.mode;
    exploration_value_ = request.value;
    return {
        .accepted = true,
        .mode = exploration_mode_,
        .value = exploration_value_,
        .matchup_counter_reset = request.reset_matchup_counter,
    };
}

ResetTransition ServerSessionV21::request_reset(
    ResetReason reason,
    std::uint64_t episode_id,
    std::uint64_t reset_id,
    std::int64_t timestamp_ms,
    bool terminal_committed
) {
    ResetTransition result{
        .state = reset_state_,
        .reset_id = active_reset_id_,
        .episode_id = episode_id,
    };
    if (reset_id == 0 || episode_id == 0 || timestamp_ms <= 0 ||
        reason == ResetReason::none) {
        result.message = "invalid reset request";
        return result;
    }
    if (active_reset_id_ == reset_id &&
        reset_state_ != ResetState::idle) {
        result.accepted = true;
        result.duplicate = true;
        result.state = reset_state_;
        return result;
    }
    if (decisions_blocked()) {
        result.message = "another reset is already active";
        return result;
    }
    if (reset_requires_committed_terminal(reason) && !terminal_committed) {
        result.message = "terminal must be committed before reset";
        return result;
    }
    active_reset_id_ = reset_id;
    reset_episode_id_ = episode_id;
    reset_reason_ = reason;
    reset_state_ = ResetState::requested;
    reset_requested_timestamp_ms_ = timestamp_ms;
    reset_acknowledged_timestamp_ms_ = 0;
    result.accepted = true;
    result.episode_aborted = !reset_requires_committed_terminal(reason);
    result.state = reset_state_;
    result.reset_id = active_reset_id_;
    return result;
}

ResetTransition ServerSessionV21::acknowledge_reset(
    const ResetAck& acknowledgement
) {
    ResetTransition result{
        .state = reset_state_,
        .reset_id = active_reset_id_,
        .episode_id = reset_episode_id_,
    };
    if (acknowledgement.reset_id != active_reset_id_ ||
        acknowledgement.episode_id != reset_episode_id_) {
        result.message = "reset acknowledgement does not match active reset";
        return result;
    }
    if (reset_state_ == ResetState::waiting_stable_state ||
        reset_state_ == ResetState::completed) {
        result.accepted = true;
        result.duplicate = true;
        return result;
    }
    if (reset_state_ != ResetState::requested &&
        reset_state_ != ResetState::command_sent &&
        reset_state_ != ResetState::client_executing) {
        result.message = "reset acknowledgement is out of order";
        return result;
    }
    if (!acknowledgement.command_executed) {
        reset_state_ = ResetState::failed;
        result.message = "client did not execute reset command";
    } else {
        reset_state_ = ResetState::waiting_stable_state;
        reset_acknowledged_timestamp_ms_ =
            acknowledgement.metadata.timestamp_ms;
        result.accepted = true;
    }
    result.state = reset_state_;
    return result;
}

ResetTransition ServerSessionV21::mark_reset_command_sent(
    std::uint64_t reset_id
) {
    ResetTransition result{
        .state = reset_state_,
        .reset_id = active_reset_id_,
        .episode_id = reset_episode_id_,
    };
    if (reset_id != active_reset_id_) {
        result.message = "reset command does not match active reset";
        return result;
    }
    if (reset_state_ == ResetState::command_sent ||
        reset_state_ == ResetState::client_executing ||
        reset_state_ == ResetState::waiting_stable_state) {
        result.accepted = true;
        result.duplicate = true;
        return result;
    }
    if (reset_state_ != ResetState::requested) {
        result.message = "reset command is out of order";
        return result;
    }
    reset_state_ = ResetState::command_sent;
    result.accepted = true;
    result.state = reset_state_;
    return result;
}

ResetTransition ServerSessionV21::mark_reset_ready(
    const ResetReady& ready
) {
    ResetTransition result{
        .state = reset_state_,
        .reset_id = active_reset_id_,
        .episode_id = reset_episode_id_,
    };
    if (ready.reset_id != active_reset_id_) {
        result.message = "reset_ready does not match active reset";
        return result;
    }
    if (reset_state_ == ResetState::completed) {
        result.accepted = true;
        result.duplicate = true;
        result.new_episode_id = episode_id_;
        return result;
    }
    if (reset_state_ != ResetState::waiting_stable_state ||
        !ready.stable_initial_state || ready.first_stable_frame_id == 0) {
        result.message = "reset_ready is out of order or state is unstable";
        return result;
    }
    const std::uint64_t previous = std::max(episode_id_, reset_episode_id_);
    episode_id_ = previous + 1U;
    episode_active_ = true;
    encounter_.current_stage = 0;
    last_stage_transition_id_ = 0;
    ignore_opponent_health_delta_ = false;
    apply_pending_mode();
    reset_state_ = ResetState::completed;
    result.accepted = true;
    result.state = reset_state_;
    result.new_episode_id = episode_id_;
    return result;
}

ResetTransition ServerSessionV21::mark_reset_failed(
    const ResetFailed& failure
) {
    ResetTransition result{
        .state = reset_state_,
        .reset_id = active_reset_id_,
        .episode_id = reset_episode_id_,
    };
    if (failure.reset_id != active_reset_id_) {
        result.message = "reset_failed does not match active reset";
        return result;
    }
    reset_state_ = ResetState::failed;
    result.accepted = true;
    result.state = reset_state_;
    result.message = failure.reason;
    return result;
}

ResetTransition ServerSessionV21::check_reset_timeout(
    std::int64_t now_ms,
    std::int64_t acknowledgement_timeout_ms,
    std::int64_t ready_timeout_ms
) {
    ResetTransition result{
        .state = reset_state_,
        .reset_id = active_reset_id_,
        .episode_id = reset_episode_id_,
    };
    bool timed_out = false;
    if ((reset_state_ == ResetState::requested ||
         reset_state_ == ResetState::command_sent ||
         reset_state_ == ResetState::client_executing) &&
        reset_requested_timestamp_ms_ > 0 &&
        now_ms - reset_requested_timestamp_ms_ >
            acknowledgement_timeout_ms) {
        timed_out = true;
        result.message = "reset_ack timeout";
    } else if (reset_state_ == ResetState::waiting_stable_state &&
        reset_acknowledged_timestamp_ms_ > 0 &&
        now_ms - reset_acknowledged_timestamp_ms_ > ready_timeout_ms) {
        timed_out = true;
        result.message = "reset_ready timeout";
    }
    if (timed_out) {
        reset_state_ = ResetState::failed;
        result.accepted = true;
        result.state = reset_state_;
    }
    return result;
}

const EncounterConfiguration& ServerSessionV21::encounter() const {
    return encounter_;
}

AgentMode ServerSessionV21::mode() const { return mode_; }
std::optional<AgentMode> ServerSessionV21::pending_mode() const {
    return pending_mode_;
}
ExplorationMode ServerSessionV21::exploration_mode() const {
    return exploration_mode_;
}
float ServerSessionV21::exploration_value() const {
    return exploration_value_;
}
float ServerSessionV21::epsilon_minimum() const { return epsilon_minimum_; }
bool ServerSessionV21::episode_active() const { return episode_active_; }
std::uint64_t ServerSessionV21::episode_id() const { return episode_id_; }
ResetState ServerSessionV21::reset_state() const { return reset_state_; }
std::uint64_t ServerSessionV21::active_reset_id() const {
    return active_reset_id_;
}
ResetReason ServerSessionV21::reset_reason() const { return reset_reason_; }
bool ServerSessionV21::decisions_blocked() const {
    return reset_state_ != ResetState::idle &&
        reset_state_ != ResetState::completed;
}
bool ServerSessionV21::consume_health_delta_ignore() {
    return std::exchange(ignore_opponent_health_delta_, false);
}

}  // namespace gameagent
