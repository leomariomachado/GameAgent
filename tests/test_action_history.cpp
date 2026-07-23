#include "ActionHistory.h"

#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        gameagent::ActionHistory history(8);
        history.record_sent(1, 100, {9, 3}, 1'000);
        const auto first = history.acknowledge(
            gameagent::ActionAcknowledgement{
                .decision_id = 1,
                .applied_timestamp_ms = 1'020,
                .applied_commands = {3, 9},
            },
            1'025
        );
        require(first.accepted, "matching acknowledgement was rejected");
        require(first.commands_match, "command-set comparison must ignore ordering");
        require(first.server_roundtrip_ms == 25, "server round-trip metric is incorrect");
        require(first.applied.command_state[3] == 1.0f &&
                    first.applied.command_state[9] == 1.0f,
                "applied multi-hot command state is incorrect");

        const auto duplicate = history.acknowledge(
            gameagent::ActionAcknowledgement{
                .decision_id = 1,
                .applied_timestamp_ms = 1'030,
                .applied_commands = {9, 3},
            },
            1'035
        );
        require(!duplicate.accepted, "duplicate acknowledgement was accepted");

        history.record_sent(2, 101, {9}, 1'050);
        const auto changed = history.acknowledge(
            gameagent::ActionAcknowledgement{
                .decision_id = 2,
                .applied_timestamp_ms = 1'060,
                .applied_commands = {},
            },
            1'065
        );
        require(changed.accepted, "empty applied state was rejected");
        require(!changed.commands_match, "different applied state was reported as matching");
        require(history.last_applied_decision_id() == 2,
                "last applied decision ID is incorrect");

        const auto latest = history.latest_applied(8);
        require(latest.size() == 2, "applied history size is incorrect");
        require(latest.back().commands.empty(), "empty applied state was not retained");

        std::cout << "[PASS] GameAgent applied action history test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
