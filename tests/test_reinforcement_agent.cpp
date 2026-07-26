#include "ReinforcementAgent.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        "gameagent_reinforcement_test";
    const std::filesystem::path latest = directory / "latest.garl";
    const std::filesystem::path best = directory / "best.garl";
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
            .training_enabled = true,
            .resume = false,
            .save_replay = true,
            .checkpoint_path = latest,
            .best_checkpoint_path = best,
        };
        gameagent::ReinforcementAgent agent(config);
        require(agent.option_names().size() >= 20, "RL option set is incomplete");

        const std::vector<float> state_a{
            0.1f, -0.2f, 0.3f, -0.4f, 0.5f, 0.6f, 0.7f, 0.8f
        };
        const std::vector<float> state_b{
            0.2f, -0.1f, 0.4f, -0.3f, 0.6f, 0.5f, 0.8f, 0.7f
        };
        auto decision = agent.step({
            .state = state_a,
            .action_allowed = true,
            .episode_started = true,
            .episode_id = 1,
            .frame_id = 1,
            .decision_id = 1,
        });
        require(decision.option_steps > 0, "RL agent did not select an option");

        std::uint64_t decision_id = 2;
        while (decision.option_step < decision.option_steps) {
            decision = agent.step({
                .state = state_a,
                .reward = 0.05f,
                .action_allowed = true,
                .episode_id = 1,
                .frame_id = decision_id,
                .decision_id = decision_id,
            });
            ++decision_id;
            require(decision_id < 100, "RL option did not terminate");
        }
        decision = agent.step({
            .state = state_b,
            .reward = 0.5f,
            .action_allowed = true,
            .episode_id = 1,
            .frame_id = decision_id,
            .decision_id = decision_id,
        });
        require(decision.training_performed, "Double DQN did not train from replay");
        require(decision.replay_size >= 1, "completed option was not stored in replay");
        require(
            std::isfinite(decision.training_loss),
            "Double DQN produced a non-finite loss"
        );

        decision = agent.step({
            .state = state_b,
            .reward = 1.0f,
            .action_allowed = false,
            .episode_ended = true,
            .terminal = true,
            .episode_id = 1,
            .frame_id = decision_id + 1,
            .decision_id = decision_id + 1,
        });
        require(decision.commands.empty(), "terminal state emitted controller commands");
        require(decision.checkpoint_saved, "episode end did not save a checkpoint");
        require(
            std::filesystem::is_regular_file(latest) &&
                std::filesystem::is_regular_file(best),
            "latest/best RL checkpoints are missing"
        );
        const auto before = agent.stats();
        require(before.episodes_completed == 1, "episode counter is incorrect");

        config.resume = true;
        gameagent::ReinforcementAgent resumed(config);
        const auto after = resumed.stats();
        require(after.checkpoint_loaded, "RL checkpoint was not resumed");
        require(
            after.training_updates == before.training_updates &&
                after.episodes_completed == before.episodes_completed &&
                after.replay_size == before.replay_size,
            "RL checkpoint did not restore training state"
        );

        std::cout
            << "[INFO] options=" << resumed.option_names().size()
            << " replay=" << after.replay_size
            << " updates=" << after.training_updates
            << " epsilon=" << after.epsilon
            << " best_return=" << after.best_episode_return << '\n'
            << "[PASS] semi-Markov Double DQN/replay/checkpoint test passed\n";
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
