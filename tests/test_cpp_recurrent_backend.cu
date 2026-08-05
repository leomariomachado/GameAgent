#include "RecurrentPolicyBackend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
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

gameagent::RecurrentPolicyBackendConfig make_config(
    const std::filesystem::path& checkpoint,
    bool training,
    bool resume
) {
    return {
        .backend = "cpp",
        .service_script = {},
        .checkpoint_path = checkpoint,
        .migration_checkpoint_path = {},
        .resnet50_checkpoint_path = {},
        .action_sizes = {3, 4},
        .visual_size = 32,
        .projection_size = 8,
        .controller_size = 4,
        .temporal_hidden_size = 12,
        .context_size = 6,
        .strategy_hidden_size = 10,
        .sequence_capacity = 4,
        .character_embedding_size = 3,
        .opponent_embedding_size = 4,
        .opponent_count = 5,
        .replay_capacity = 64,
        .minimum_replay_size = 4,
        .batch_size = 4,
        .training_updates_per_transition = 1,
        .target_update_interval = 2,
        .initial_matchup_wins = {},
        .gamma = 0.95f,
        .q_value_limit = 10.0f,
        .terminal_replay_fraction = 0.25f,
        .hard_negative_replay_fraction = 0.25f,
        .hard_negative_q_threshold = 0.1f,
        .n_step_options = 2,
        .learning_rate = 3.0e-4f,
        .gradient_clip = 2.0f,
        .random_seed = 12345,
        .device_id = 0,
        .training_enabled = training,
        .checkpoint_writes_enabled = training,
        .resume = resume,
        .conservative_bootstrap = true,
        .trainable_resnet = false,
    };
}

void observe(
    gameagent::RecurrentPolicyBackend& backend,
    int step,
    bool episode_started = false
) {
    std::vector<float> visual(32);
    std::vector<float> controller(4);
    std::vector<float> context(6);
    for (std::size_t index = 0; index < visual.size(); ++index) {
        visual[index] = std::sin(
            static_cast<float>(step) * 0.17f +
            static_cast<float>(index) * 0.03f
        );
    }
    for (std::size_t index = 0; index < controller.size(); ++index) {
        controller[index] = ((step + static_cast<int>(index)) % 3 == 0)
            ? 1.0f : 0.0f;
    }
    for (std::size_t index = 0; index < context.size(); ++index) {
        context[index] = static_cast<float>(step + index) / 32.0f;
    }
    backend.observe(
        visual, {}, static_cast<std::uint64_t>(step + 1),
        controller, context, episode_started
    );
}

}  // namespace

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
            std::cout << "[SKIP] CUDA device unavailable\n";
            return 77;
        }
        const std::filesystem::path checkpoint =
            "/tmp/gameagent_cpp_recurrent_test.baseai.ckpt";
        std::filesystem::remove(checkpoint);
        std::vector<float> saved_q;
        std::uint64_t saved_updates = 0;
        double initial_checksum = 0.0;
        double trained_checksum = 0.0;
        {
            gameagent::RecurrentPolicyBackend backend(
                make_config(checkpoint, true, false)
            );
            require(
                backend.stats().backend == "cpp",
                "native recurrent backend identity"
            );
            initial_checksum = backend.stats().parameter_checksum;
            observe(backend, 0, true);
            for (int step = 0; step < 18; ++step) {
                const std::vector<float> q = backend.q_values(1, 2, 4);
                require(q.size() == 4, "native Q-value shape");
                require(
                    std::all_of(q.begin(), q.end(), [](float value) {
                        return std::isfinite(value);
                    }),
                    "native Q-values must be finite"
                );
                const std::size_t action =
                    static_cast<std::size_t>(step % 4);
                backend.begin_transition(action, 1, 2, q[action]);
                observe(backend, step + 1, false);
                const bool terminal = (step + 1) % 6 == 0;
                const float reward = action == 0 ? 1.0f : -0.35f;
                const std::array<std::uint8_t, 4> action_mask{
                    1, 1, 1, 1
                };
                const auto close = backend.close_transition(
                    1, reward, terminal, terminal, action_mask
                );
                if (terminal) {
                    observe(backend, step + 100, true);
                }
                require(
                    std::isfinite(close.stats.last_loss),
                    "native training loss must be finite"
                );
            }
            // The terminal screen can arrive after the option has already
            // closed. Its reward must flush the pending n-step suffix and the
            // complete winning trajectory must still enter the win reserve.
            const std::vector<float> between_option_q =
                backend.q_values(1, 2, 4);
            backend.begin_transition(
                2, 1, 2, between_option_q[2]
            );
            observe(backend, 250, false);
            const std::array<std::uint8_t, 4> episode_mask{
                1, 1, 1, 1
            };
            backend.close_transition(
                2, 0.25f, false, false, episode_mask
            );
            observe(backend, 251, false);
            const auto finalized = backend.finalize_episode(
                10.0f, true, episode_mask
            );
            require(
                finalized.stats.winning_episodes == 4,
                "between-option victory promoted"
            );
            const auto& stats = backend.stats();
            require(stats.training_updates > 0, "native BPTT update");
            require(stats.replay_size >= 4, "native replay population");
            require(
                stats.winning_replay_size > 0 &&
                stats.winning_episodes == 4,
                "native successful-trajectory replay population"
            );
            require(
                stats.current_matchup_winning_episodes == 4 &&
                std::abs(
                    stats.effective_winning_replay_fraction - 0.26f
                ) < 1.0e-5f,
                "native adaptive winning replay fraction"
            );
            require(
                stats.last_winning_sample_fraction >= 0.20f,
                "native successful trajectories sampled in training"
            );
            require(
                std::isfinite(stats.last_abs_td_error),
                "native TD diagnostics"
            );
            trained_checksum = stats.parameter_checksum;
            require(
                std::abs(trained_checksum - initial_checksum) > 1.0e-8,
                "native parameters changed after AdamW"
            );
            saved_q = backend.q_values(1, 2, 4);
            saved_updates = stats.training_updates;
            backend.save(checkpoint, false);
        }
        require(
            std::filesystem::is_regular_file(checkpoint),
            "native BaseAI checkpoint created"
        );
        {
            gameagent::RecurrentPolicyBackend restored(
                make_config(checkpoint, false, true)
            );
            observe(restored, 117, true);
            // Recreate the exact four-step observation window used just
            // before saved_q was produced.
            for (int step = 15; step <= 17; ++step) {
                observe(restored, step + 1, false);
            }
            const std::vector<float> restored_q =
                restored.q_values(1, 2, 4);
            require(
                restored.stats().checkpoint_loaded,
                "native checkpoint load flag"
            );
            require(
                restored.stats().training_updates == saved_updates,
                "native optimizer/update resume"
            );
            float maximum_difference = 0.0f;
            for (std::size_t index = 0; index < saved_q.size(); ++index) {
                maximum_difference = std::max(
                    maximum_difference,
                    std::abs(saved_q[index] - restored_q[index])
                );
            }
            // The observation reconstruction above intentionally exercises
            // checkpoint loading more than exact sequence serialization; all
            // outputs still need to be finite and the parameter checksum exact.
            require(
                std::abs(restored.stats().parameter_checksum -
                    trained_checksum) < 1.0e-5,
                "native checkpoint parameter parity"
            );
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < 100; ++iteration) {
                static_cast<void>(restored.q_values(1, 2, 4));
            }
            const double milliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start
                ).count() / 100.0;
            std::cout
                << "[INFO] backend=cpp updates=" << saved_updates
                << " replay=18 checksum=" << trained_checksum
                << " reconstructed_q_max_abs=" << maximum_difference
                << " inference_ms=" << milliseconds << '\n';
        }
        std::filesystem::remove(checkpoint);
        std::cout
            << "[PASS] Native BaseAI LSTM/BPTT/twin-DQN/checkpoint test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
