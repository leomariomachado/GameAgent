#include "MockAgent.h"

#include <algorithm>
#include <cmath>
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
        throw std::runtime_error("missing character move: " + name);
    }
    return *found;
}

std::vector<std::vector<int>> mirrored_program(
    std::vector<std::vector<int>> states,
    int forward,
    int back
) {
    for (auto& state : states) {
        for (int& command : state) {
            if (command == forward) {
                command = back;
            } else if (command == back) {
                command = forward;
            }
        }
    }
    return states;
}

void require_mirrored_pair(
    const std::vector<gameagent::MockMoveDefinition>& moves,
    const std::string& original_name,
    int forward,
    int back
) {
    const auto& original = find_move(moves, original_name);
    const auto& mirror = find_move(
        moves, original_name + " [Espelho F/B]"
    );
    require(
        mirror.states ==
            mirrored_program(original.states, forward, back),
        "controller states were not spatially mirrored: " + original_name
    );
    require(
        mirror.phases == original.phases &&
            mirror.action_source == original.action_source &&
            mirror.curriculum_weight == original.curriculum_weight,
        "mirror metadata/weight differs from original: " + original_name
    );
}

}  // namespace

int main() {
    try {
        require(
            gameagent::parse_fighter_character("Sub-Zero") ==
                gameagent::FighterCharacter::subzero &&
                gameagent::parse_fighter_character("noob saibot") ==
                    gameagent::FighterCharacter::noob_saibot &&
                !gameagent::parse_fighter_character("unknown").has_value(),
            "fighter character parsing is incorrect"
        );
        const auto supported_opponents =
            gameagent::supported_opponent_characters();
        require(
            supported_opponents.size() ==
                gameagent::kOpponentCharacterFeatureCount &&
                gameagent::parse_opponent_character("Shao Kahn") ==
                    gameagent::OpponentCharacter::shao_kahn &&
                gameagent::parse_opponent_character("Motaro") ==
                    gameagent::OpponentCharacter::motaro &&
                gameagent::parse_opponent_character("Montaro") ==
                    gameagent::OpponentCharacter::motaro &&
                gameagent::parse_opponent_character("Smoke (Human)") ==
                    gameagent::OpponentCharacter::human_smoke &&
                gameagent::parse_opponent_character("Sub-Zero Classic") ==
                    gameagent::OpponentCharacter::classic_subzero &&
                gameagent::parse_opponent_character("Sonya Blade") ==
                    gameagent::OpponentCharacter::sonya &&
                !gameagent::parse_opponent_character("invalid").has_value(),
            "UMK3 opponent character parsing is incorrect"
        );
        require(
            gameagent::opponent_character_slug(
                gameagent::OpponentCharacter::shao_kahn
            ) == "shao_kahn" &&
                gameagent::opponent_character_name(
                    gameagent::OpponentCharacter::motaro
                ) == "Motaro" &&
                std::find(
                    supported_opponents.begin(),
                    supported_opponents.end(),
                    "sheeva"
                ) != supported_opponents.end(),
            "UMK3 opponent character vocabulary is incomplete"
        );

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
            curriculum.size() == 9,
            "Sub-Zero must use only its six base moves and three unique mirrors"
        );
        require(
            find_move(curriculum, "Congelar").states ==
                mirrored_program(
                    find_move(curriculum, "Clone de Gelo").states, 9, 8
                ),
            "Freeze and Ice Clone must remain the existing mirrored pair"
        );
        const std::vector<std::string> spatial_programs{
            "Chuva de Gelo (Frente)",
            "Chuva de Gelo (Perto)",
            "4 Golpes",
        };
        for (const auto& name : spatial_programs) {
            require_mirrored_pair(curriculum, name, 9, 8);
        }
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
            std::abs(total_curriculum_weight - 100.0f) < 0.001f,
            "Sub-Zero curriculum weights must add up to 100"
        );

        const auto noob = gameagent::make_character_curriculum(
            gameagent::FighterCharacter::noob_saibot,
            gameagent::FighterFacing::right
        );
        require(noob.size() == 7, "Noob Saibot curriculum size is incorrect");
        require(
            find_move(noob, "Arremesso de Sombra").states ==
                std::vector<std::vector<int>>{
                    {9}, {}, {9}, {0}, {}, {}
                } &&
                find_move(noob, "Teleporte com Arremesso").states ==
                    std::vector<std::vector<int>>{
                        {7}, {6}, {}, {}
                    } &&
                find_move(noob, "Nuvem de Desativação").states ==
                    std::vector<std::vector<int>>{
                        {7}, {9}, {2}, {}, {}
                    },
            "Noob Saibot commands are incorrect"
        );
        require_mirrored_pair(noob, "Arremesso de Sombra", 9, 8);
        require_mirrored_pair(noob, "Nuvem de Desativação", 9, 8);

        const auto scorpion = gameagent::make_character_curriculum(
            gameagent::FighterCharacter::scorpion,
            gameagent::FighterFacing::right
        );
        require(scorpion.size() == 11, "Scorpion curriculum size is incorrect");
        require(
            find_move(scorpion, "Arpão").states ==
                std::vector<std::vector<int>>{
                    {8}, {}, {8}, {2}, {}, {}
                } &&
                find_move(scorpion, "Soco Teleportado").states ==
                    std::vector<std::vector<int>>{
                        {7}, {8}, {0}, {}, {}
                    } &&
                find_move(scorpion, "Agarrão Aéreo").states ==
                    std::vector<std::vector<int>>{
                        {6}, {6}, {}, {4}, {4}, {}, {}
                    },
            "Scorpion commands are incorrect"
        );
        require_mirrored_pair(scorpion, "Arpão", 9, 8);
        require_mirrored_pair(scorpion, "Soco Teleportado", 9, 8);

        const auto sindel = gameagent::make_character_curriculum(
            gameagent::FighterCharacter::sindel,
            gameagent::FighterFacing::right
        );
        require(sindel.size() == 12, "Sindel curriculum size is incorrect");
        require(
            find_move(sindel, "Bola de Fogo").states ==
                std::vector<std::vector<int>>{
                    {9}, {}, {9}, {2}, {}, {}
                } &&
                find_move(sindel, "Grito").states ==
                    std::vector<std::vector<int>>{
                        {9}, {}, {9}, {}, {9}, {0}, {}, {}
                    } &&
                find_move(sindel, "Levitar").states ==
                    std::vector<std::vector<int>>{
                        {8}, {}, {8}, {9}, {1}, {}, {}
                    },
            "Sindel commands are incorrect"
        );
        require_mirrored_pair(sindel, "Bola de Fogo", 9, 8);
        require_mirrored_pair(sindel, "Levitar", 9, 8);

        for (const auto* profile :
             {&curriculum, &noob, &scorpion, &sindel}) {
            float weight = 0.0f;
            for (const auto& move : *profile) {
                weight += move.curriculum_weight;
                require(
                    move.states.size() == move.phases.size(),
                    "character profile state/phase counts differ"
                );
            }
            require(
                std::abs(weight - 100.0f) < 0.001f,
                "character curriculum weights must add up to 100"
            );
        }

        const auto left_curriculum = gameagent::make_sub_zero_curriculum(
            gameagent::FighterFacing::left
        );
        for (const auto& name : spatial_programs) {
            require_mirrored_pair(left_curriculum, name, 8, 9);
        }
        require(
            find_move(left_curriculum, "4 Golpes").states[4] ==
                    std::vector<int>({9, 1}) &&
                find_move(left_curriculum, "4 Golpes").states[6] ==
                    std::vector<int>({8, 3}),
            "Sub-Zero curriculum did not map relative directions when facing left"
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

        std::cout
            << "[PASS] homogeneous multi-character mock curricula test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
