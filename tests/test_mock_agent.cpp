#include "MockAgent.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const gameagent::MockMoveDefinition& find_move(
    const std::vector<gameagent::MockMoveDefinition>& moves,
    const std::string& name
) {
    const auto found = std::find_if(
        moves.begin(), moves.end(), [&](const gameagent::MockMoveDefinition& move) {
            return move.name == name;
        }
    );
    if (found == moves.end()) {
        throw std::runtime_error("missing Sub-Zero move: " + name);
    }
    return *found;
}

}  // namespace

int main() {
    try {
        const auto right_facing = gameagent::make_sub_zero_mock_moves(
            gameagent::FighterFacing::right
        );
        require(right_facing.size() == 6, "expected exactly six Sub-Zero moves");
        require(
            find_move(right_facing, "Congelar").states ==
                std::vector<std::vector<int>>{{7}, {9}, {2}, {}, {}},
            "Congelar sequence is incorrect"
        );
        require(
            find_move(right_facing, "Clone de Gelo").states ==
                std::vector<std::vector<int>>{{7}, {8}, {2}, {}, {}},
            "Clone de Gelo sequence is incorrect"
        );
        require(
            find_move(right_facing, "Chuva de Gelo (Frente)").states ==
                std::vector<std::vector<int>>{{7}, {9}, {2}, {0}, {}, {}},
            "front Ice Shower sequence is incorrect"
        );
        require(
            find_move(right_facing, "Chuva de Gelo (Perto)").states ==
                std::vector<std::vector<int>>{{7}, {8}, {9}, {0}, {}, {}},
            "near Ice Shower sequence is incorrect"
        );
        require(
            find_move(right_facing, "3 Golpes").states ==
                std::vector<std::vector<int>>{
                    {0}, {}, {0}, {}, {7, 2}, {}, {}
                },
            "three-hit sequence is incorrect"
        );
        require(
            find_move(right_facing, "4 Golpes").states ==
                std::vector<std::vector<int>>{
                    {1}, {}, {1}, {}, {8, 1}, {}, {9, 3}, {}, {}
                },
            "four-hit sequence is incorrect"
        );

        const auto left_facing = gameagent::make_sub_zero_mock_moves(
            gameagent::FighterFacing::left
        );
        require(
            find_move(left_facing, "Clone de Gelo").states ==
                std::vector<std::vector<int>>{{7}, {9}, {2}, {}, {}},
            "left-facing back direction was not mirrored"
        );
        require(
            find_move(left_facing, "4 Golpes").states[4] ==
                std::vector<int>{9, 1} &&
                find_move(left_facing, "4 Golpes").states[6] ==
                    std::vector<int>{8, 3},
            "left-facing simultaneous directions were not mirrored"
        );

        auto frame = std::make_shared<gameagent::BufferedFrame>();
        frame->frame_id = 1;
        const std::vector<std::shared_ptr<const gameagent::BufferedFrame>> observation{
            frame
        };
        const std::vector<gameagent::AppliedActionRecord> history;
        gameagent::MockAgent agent(12345U, gameagent::FighterFacing::right);

        gameagent::MockAgentDecision decision = agent.infer(observation, history);
        const auto& selected = find_move(right_facing, decision.move_name);
        for (std::size_t step = 0; step < selected.states.size(); ++step) {
            if (step != 0) {
                decision = agent.infer(observation, history);
            }
            require(decision.move_name == selected.name,
                    "mock changed move before completing its sequence");
            require(decision.step_index == step + 1U,
                    "mock returned the wrong sequence step");
            require(decision.step_count == selected.states.size(),
                    "mock returned the wrong sequence length");
            require(decision.commands == selected.states[step],
                    "mock returned commands outside the selected move");
        }

        const auto next = agent.infer(observation, history);
        require(next.step_index == 1U, "next move did not restart at step one");
        agent.reset_sequence();
        const auto reset = agent.infer(observation, history);
        require(reset.step_index == 1U, "connection reset did not restart the move");

        std::cout << "[PASS] Sub-Zero temporal mock agent test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
