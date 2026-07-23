#include "ActionHistory.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gameagent {
namespace {

std::array<float, 10> make_command_state(const std::vector<int>& commands) {
    std::array<float, 10> state{};
    for (const int command : commands) {
        if (command < 0 || command >= static_cast<int>(state.size())) {
            throw std::invalid_argument("command is outside the 0..9 range");
        }
        float& value = state[static_cast<std::size_t>(command)];
        if (value != 0.0f) {
            throw std::invalid_argument("commands must be unique");
        }
        value = 1.0f;
    }
    return state;
}

}  // namespace

ActionHistory::ActionHistory(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::invalid_argument("action history capacity must be positive");
    }
}

void ActionHistory::record_sent(
    std::uint64_t decision_id,
    std::uint64_t based_on_frame_id,
    const std::vector<int>& commands,
    std::int64_t generated_timestamp_ms
) {
    if (decision_id == 0 || generated_timestamp_ms <= 0) {
        throw std::invalid_argument("sent action has invalid identifiers or timestamp");
    }
    make_command_state(commands);

    std::lock_guard lock(mutex_);
    if (decision_id <= last_sent_decision_id_) {
        throw std::invalid_argument("sent decision_id must increase monotonically");
    }
    sent_.push_back(SentAction{
        .decision_id = decision_id,
        .based_on_frame_id = based_on_frame_id,
        .commands = commands,
        .generated_timestamp_ms = generated_timestamp_ms,
    });
    last_sent_decision_id_ = decision_id;
    while (sent_.size() > capacity_) {
        sent_.pop_front();
    }
}

ActionAcknowledgementResult ActionHistory::acknowledge(
    const ActionAcknowledgement& acknowledgement,
    std::int64_t received_timestamp_ms
) {
    std::lock_guard lock(mutex_);
    if (acknowledgement.decision_id <= last_applied_decision_id_) {
        return {
            .accepted = false,
            .reason = "stale or duplicate action acknowledgement",
        };
    }
    if (acknowledgement.decision_id > last_sent_decision_id_) {
        return {
            .accepted = false,
            .reason = "acknowledgement refers to a decision not sent by this connection",
        };
    }
    const auto sent = std::find_if(sent_.begin(), sent_.end(), [&](const SentAction& value) {
        return value.decision_id == acknowledgement.decision_id;
    });
    if (sent == sent_.end()) {
        return {
            .accepted = false,
            .reason = "acknowledged decision is no longer available in sent history",
        };
    }

    const auto sent_state = make_command_state(sent->commands);
    const auto applied_state = make_command_state(acknowledgement.applied_commands);
    AppliedActionRecord record{
        .decision_id = acknowledgement.decision_id,
        .based_on_frame_id = sent->based_on_frame_id,
        .commands = acknowledgement.applied_commands,
        .command_state = applied_state,
        .generated_timestamp_ms = sent->generated_timestamp_ms,
        .applied_timestamp_ms = acknowledgement.applied_timestamp_ms,
        .acknowledgement_received_timestamp_ms = received_timestamp_ms,
        .commands_match = sent_state == applied_state,
    };
    const std::int64_t server_roundtrip_ms =
        received_timestamp_ms - sent->generated_timestamp_ms;

    applied_.push_back(record);
    last_applied_decision_id_ = acknowledgement.decision_id;
    while (applied_.size() > capacity_) {
        applied_.pop_front();
    }
    while (!sent_.empty() && sent_.front().decision_id <= acknowledgement.decision_id) {
        sent_.pop_front();
    }

    return {
        .accepted = true,
        .commands_match = record.commands_match,
        .reason = {},
        .server_roundtrip_ms = server_roundtrip_ms,
        .applied = std::move(record),
    };
}

std::vector<AppliedActionRecord> ActionHistory::latest_applied(std::size_t count) const {
    std::lock_guard lock(mutex_);
    const std::size_t selected = std::min(count, applied_.size());
    const auto begin = applied_.end() - static_cast<std::ptrdiff_t>(selected);
    return {begin, applied_.end()};
}

std::size_t ActionHistory::applied_size() const {
    std::lock_guard lock(mutex_);
    return applied_.size();
}

std::uint64_t ActionHistory::last_applied_decision_id() const {
    std::lock_guard lock(mutex_);
    return last_applied_decision_id_;
}

}  // namespace gameagent
