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

        const auto curriculum = gameagent::make_sub_zero_curriculum(
            gameagent::FighterFacing::right
        );
        require(
            curriculum.size() == 17,
            "Sub-Zero curriculum must contain 6 moves, 2 conditional plans, "
            "4 combat variations and 5 explorations"
        );
        const auto& freeze_combo = find_move(
            curriculum, "Congelar + Corrida + 3 Golpes"
        );
        require(
            freeze_combo.conditional_freeze_follow_up &&
                freeze_combo.curriculum_weight == 25.0f &&
                freeze_combo.states[7] == std::vector<int>({5, 9}) &&
                freeze_combo.states[15] == std::vector<int>({7, 2}),
            "conditional freeze/approach/combo plan is incorrect"
        );
        const auto& jump_kick = find_move(
            curriculum, "Pulo para Frente + Chute Alto"
        );
        require(
            jump_kick.states ==
                std::vector<std::vector<int>>{
                    {6, 9}, {6, 9}, {9}, {9, 1}, {9, 1}, {}, {}
                } &&
                jump_kick.action_source == "scripted_variation" &&
                jump_kick.curriculum_weight == 4.0f,
            "forward jump/high kick variation is incorrect"
        );
        const auto& uppercut = find_move(curriculum, "Uppercut Agachado");
        require(
            uppercut.states ==
                std::vector<std::vector<int>>{
                    {7}, {7}, {7, 0}, {7, 0}, {}, {}
                } &&
                uppercut.curriculum_weight == 3.0f,
            "crouching uppercut variation is incorrect"
        );
        const auto& running_combo = find_move(
            curriculum, "Corrida + 4 Golpes"
        );
        require(
            running_combo.states[0] == std::vector<int>({5, 9}) &&
                running_combo.states[9] == std::vector<int>({8, 1}) &&
                running_combo.states[11] == std::vector<int>({9, 3}),
            "running four-hit combo variation is incorrect"
        );
        const auto& exploration = find_move(
            curriculum, "Explorar: bloquear e recuar"
        );
        require(
            exploration.action_source == "random_exploration" &&
                exploration.curriculum_weight == 1.0f,
            "controlled exploration curriculum is incorrect"
        );
        float total_curriculum_weight = 0.0f;
        for (const auto& move : curriculum) {
            total_curriculum_weight += move.curriculum_weight;
            require(!move.states.empty(), "curriculum move has no states");
            require(
                move.phases.size() == move.states.size(),
                "curriculum move phase/state counts differ: " + move.name
            );
            for (const auto& state : move.states) {
                require(
                    std::find(state.begin(), state.end(), 6) == state.end() ||
                        std::find(state.begin(), state.end(), 7) == state.end(),
                    "curriculum state presses Up and Down together: " +
                        move.name
                );
                require(
                    std::find(state.begin(), state.end(), 8) == state.end() ||
                        std::find(state.begin(), state.end(), 9) == state.end(),
                    "curriculum state presses Left and Right together: " +
                        move.name
                );
                for (std::size_t index = 0; index < state.size(); ++index) {
                    require(
                        state[index] >= 0 && state[index] <= 9,
                        "curriculum state contains an invalid command: " +
                            move.name
                    );
                    require(
                        std::find(
                            state.begin(),
                            state.begin() +
                                static_cast<std::ptrdiff_t>(index),
                            state[index]
                        ) ==
                            state.begin() +
                                static_cast<std::ptrdiff_t>(index),
                        "curriculum state contains duplicate commands: " +
                            move.name
                    );
                }
            }
        }
        require(
            total_curriculum_weight == 100.0f,
            "Sub-Zero curriculum weights must add up to 100"
        );

        const auto left_curriculum = gameagent::make_sub_zero_curriculum(
            gameagent::FighterFacing::left
        );
        require(
            find_move(
                left_curriculum, "Pulo para Frente + Chute Alto"
            ).states[0] == std::vector<int>({6, 8}) &&
                find_move(left_curriculum, "Corrida + 4 Golpes").states[0] ==
                    std::vector<int>({5, 8}) &&
                find_move(left_curriculum, "Corrida + 4 Golpes").states[9] ==
                    std::vector<int>({9, 1}) &&
                find_move(left_curriculum, "Corrida + 4 Golpes").states[11] ==
                    std::vector<int>({8, 3}),
            "new combat variations were not mirrored when facing left"
        );

        base::data::image::Image neutral_image(32, 24, 3);
        std::fill(
            neutral_image.bytes().begin(), neutral_image.bytes().end(), 0U
        );
        base::data::image::Image frozen_image = neutral_image;
        for (int y = 4; y < 20; ++y) {
            for (int x = 18; x < 30; ++x) {
                frozen_image.at(y, x, 0) = 40;
                frozen_image.at(y, x, 1) = 170;
                frozen_image.at(y, x, 2) = 230;
            }
        }
        require(
            gameagent::sub_zero_freeze_visual_score(
                frozen_image, gameagent::FighterFacing::right
            ) >
                gameagent::sub_zero_freeze_visual_score(
                    neutral_image, gameagent::FighterFacing::right
                ),
            "freeze visual detector did not react to opponent-side ice colors"
        );

        auto frame = std::make_shared<gameagent::BufferedFrame>();
        frame->frame_id = 1;
        const std::vector<std::shared_ptr<const gameagent::BufferedFrame>> observation{
            frame
        };
        const std::vector<gameagent::AppliedActionRecord> history;
        gameagent::MockAgent agent(12345U, gameagent::FighterFacing::right);

        gameagent::MockAgentDecision decision = agent.infer(observation, history);
        const auto& selected = find_move(curriculum, decision.move_name);
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
