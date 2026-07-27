#include "ReinforcementAgent.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

gameagent::ReinforcementDecision complete_one_option(
    gameagent::ReinforcementAgent& agent,
    const std::vector<float>& start,
    const std::vector<float>& next,
    std::uint64_t episode_id,
    std::uint64_t& decision_id
) {
    auto decision = agent.step({
        .state = start,
        .action_allowed = true,
        .episode_started = true,
        .episode_id = episode_id,
        .frame_id = decision_id,
        .decision_id = decision_id,
    });
    ++decision_id;
    require(
        decision.option_steps > 0,
        "shared RL agent did not select an option"
    );
    while (decision.option_step < decision.option_steps) {
        decision = agent.step({
            .state = start,
            .reward = 0.05f,
            .action_allowed = true,
            .episode_id = episode_id,
            .frame_id = decision_id,
            .decision_id = decision_id,
        });
        ++decision_id;
        require(decision_id < 200, "RL option did not terminate");
    }
    return agent.step({
        .state = next,
        .reward = 0.5f,
        .action_allowed = true,
        .episode_id = episode_id,
        .frame_id = decision_id,
        .decision_id = decision_id,
    });
}

}  // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        "gameagent_shared_reinforcement_test";
    const std::filesystem::path latest =
        directory / "shared_policy_latest.garl";
    const std::filesystem::path subzero_best =
        directory / "shared_policy_best_subzero.garl";
    const std::filesystem::path scorpion_best =
        directory / "shared_policy_best_scorpion.garl";
    try {
        std::filesystem::remove_all(directory);
        gameagent::ReinforcementAgentConfig config{
            .input_size = 8,
            .network_hidden_size = 16,
            .replay_capacity = 64,
            .minimum_replay_size = 1,
            .batch_size = 1,
            .training_updates_per_transition = 1,
            .target_update_interval = 2,
            .gamma = 0.95f,
            .learning_rate = 1.0e-3f,
            .epsilon_start = 0.0f,
            .epsilon_final = 0.0f,
            .epsilon_decay_selections = 1,
            .scripted_exploration_probability = 0.5f,
            .random_seed = 12345,
            .character = gameagent::FighterCharacter::subzero,
            .opponent_character =
                gameagent::OpponentCharacter::scorpion,
            .character_embedding_size = 4,
            .opponent_embedding_size = 4,
            .training_enabled = true,
            .resume = false,
            .save_replay = true,
            .checkpoint_path = latest,
            .best_checkpoint_path = subzero_best,
        };
        const std::vector<float> state_a{
            0.1f, -0.2f, 0.3f, -0.4f,
            0.5f, 0.6f, 0.7f, 0.8f
        };
        const std::vector<float> state_b{
            0.2f, -0.1f, 0.4f, -0.3f,
            0.6f, 0.5f, 0.8f, 0.7f
        };

        gameagent::ReinforcementAgent subzero(config);
        require(
            subzero.option_names().size() == 20,
            "Sub-Zero did not receive its action head"
        );
        const double initial_shared_checksum =
            subzero.stats().shared_parameter_checksum;
        std::uint64_t decision_id = 1;
        auto decision = complete_one_option(
            subzero, state_a, state_b, 1, decision_id
        );
        require(
            decision.training_performed &&
                decision.replay_size >= 1 &&
                std::isfinite(decision.training_loss),
            "Sub-Zero experience did not train the shared trunk"
        );
        decision = subzero.step({
            .state = state_b,
            .reward = 1.0f,
            .action_allowed = false,
            .episode_ended = true,
            .terminal = true,
            .episode_id = 1,
            .frame_id = decision_id + 1,
            .decision_id = decision_id + 1,
        });
        require(
            decision.commands.empty() && decision.checkpoint_saved,
            "terminal state did not release controls and save shared policy"
        );
        const auto subzero_stats = subzero.stats();
        require(
            subzero_stats.shared_parameter_checksum !=
                initial_shared_checksum &&
                subzero_stats.shared_replay_matchups == 1,
            "shared parameters/replay were not updated"
        );
        require(
            std::filesystem::is_regular_file(latest) &&
                std::filesystem::is_regular_file(subzero_best),
            "Sub-Zero did not write shared latest/best checkpoints"
        );

        auto scorpion_config = config;
        scorpion_config.character =
            gameagent::FighterCharacter::scorpion;
        scorpion_config.opponent_character =
            gameagent::OpponentCharacter::sindel;
        scorpion_config.facing = gameagent::FighterFacing::left;
        scorpion_config.resume = true;
        scorpion_config.best_checkpoint_path = scorpion_best;
        gameagent::ReinforcementAgent scorpion(scorpion_config);
        const auto inherited = scorpion.stats();
        require(
            inherited.checkpoint_loaded &&
                inherited.training_updates ==
                    subzero_stats.training_updates &&
                inherited.replay_size == subzero_stats.replay_size &&
                inherited.shared_parameter_checksum ==
                    subzero_stats.shared_parameter_checksum,
            "Scorpion did not inherit Sub-Zero shared knowledge"
        );
        require(
            inherited.action_selections == 0 &&
                scorpion.option_names().size() == 22,
            "Scorpion exploration/head was not kept character-specific"
        );

        decision_id = 100;
        decision = complete_one_option(
            scorpion, state_b, state_a, 2, decision_id
        );
        require(
            decision.training_performed,
            "Scorpion experience did not train shared policy"
        );
        decision = scorpion.step({
            .state = state_a,
            .reward = 0.75f,
            .action_allowed = false,
            .episode_ended = true,
            .terminal = true,
            .episode_id = 2,
            .frame_id = decision_id + 1,
            .decision_id = decision_id + 1,
        });
        const auto scorpion_stats = scorpion.stats();
        require(
            scorpion_stats.shared_replay_matchups == 2 &&
                scorpion_stats.training_updates >
                    subzero_stats.training_updates &&
                std::filesystem::is_regular_file(scorpion_best),
            "shared replay did not retain both matchups"
        );

        config.resume = true;
        gameagent::ReinforcementAgent resumed_subzero(config);
        const auto transferred_back = resumed_subzero.stats();
        require(
            transferred_back.checkpoint_loaded &&
                transferred_back.training_updates ==
                    scorpion_stats.training_updates &&
                transferred_back.replay_size ==
                    scorpion_stats.replay_size &&
                transferred_back.shared_replay_matchups == 2,
            "Sub-Zero did not receive Scorpion shared updates"
        );
        require(
            transferred_back.action_selections ==
                subzero_stats.action_selections,
            "per-character epsilon progress was not preserved"
        );

        const std::vector<
            std::pair<gameagent::FighterCharacter, std::size_t>
        > heads{
            {gameagent::FighterCharacter::subzero, 20U},
            {gameagent::FighterCharacter::noob_saibot, 18U},
            {gameagent::FighterCharacter::scorpion, 22U},
            {gameagent::FighterCharacter::sindel, 23U},
        };
        for (const auto& [character, expected_options] : heads) {
            auto head_config = config;
            head_config.character = character;
            head_config.opponent_character =
                gameagent::OpponentCharacter::shao_kahn;
            gameagent::ReinforcementAgent character_agent(head_config);
            require(
                character_agent.stats().checkpoint_loaded &&
                    character_agent.option_names().size() ==
                        expected_options,
                "shared checkpoint did not expose the requested action head"
            );
        }

        std::cout
            << "[INFO] shared_updates="
            << transferred_back.training_updates
            << " shared_replay="
            << transferred_back.replay_size
            << " matchups="
            << transferred_back.shared_replay_matchups
            << " subzero_epsilon="
            << transferred_back.epsilon
            << " scorpion_epsilon="
            << scorpion_stats.epsilon << '\n'
            << "[PASS] shared multi-character Double DQN test passed\n";
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
