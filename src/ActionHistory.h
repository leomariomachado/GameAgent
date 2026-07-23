#pragma once

#include "StreamProtocol.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace gameagent {

struct AppliedActionRecord {
    std::uint64_t decision_id = 0;
    std::uint64_t based_on_frame_id = 0;
    std::vector<int> commands;
    std::array<float, 10> command_state{};
    std::int64_t generated_timestamp_ms = 0;
    std::int64_t applied_timestamp_ms = 0;
    std::int64_t acknowledgement_received_timestamp_ms = 0;
    bool commands_match = false;
};

struct ActionAcknowledgementResult {
    bool accepted = false;
    bool commands_match = false;
    std::string reason;
    std::int64_t server_roundtrip_ms = -1;
    AppliedActionRecord applied;
};

class ActionHistory {
public:
    explicit ActionHistory(std::size_t capacity);

    void record_sent(
        std::uint64_t decision_id,
        std::uint64_t based_on_frame_id,
        const std::vector<int>& commands,
        std::int64_t generated_timestamp_ms
    );

    ActionAcknowledgementResult acknowledge(
        const ActionAcknowledgement& acknowledgement,
        std::int64_t received_timestamp_ms
    );

    [[nodiscard]] std::vector<AppliedActionRecord> latest_applied(
        std::size_t count
    ) const;
    [[nodiscard]] std::size_t applied_size() const;
    [[nodiscard]] std::uint64_t last_applied_decision_id() const;

private:
    struct SentAction {
        std::uint64_t decision_id = 0;
        std::uint64_t based_on_frame_id = 0;
        std::vector<int> commands;
        std::int64_t generated_timestamp_ms = 0;
    };

    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<SentAction> sent_;
    std::deque<AppliedActionRecord> applied_;
    std::uint64_t last_sent_decision_id_ = 0;
    std::uint64_t last_applied_decision_id_ = 0;
};

}  // namespace gameagent
