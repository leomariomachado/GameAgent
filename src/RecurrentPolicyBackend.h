#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gameagent {

struct RecurrentPolicyBackendConfig {
    // Native C++/CUDA is the production default.  The Python sidecar remains
    // available as a compatibility/reference backend while checkpoints are
    // migrated and numerical parity is verified.
    std::string backend = "cpp";
    std::filesystem::path service_script;
    std::filesystem::path checkpoint_path;
    std::filesystem::path migration_checkpoint_path;
    std::filesystem::path resnet50_checkpoint_path;
    std::vector<std::size_t> action_sizes;
    std::size_t visual_size = 2048;
    std::size_t projection_size = 256;
    std::size_t controller_size = 12;
    std::size_t temporal_hidden_size = 256;
    std::size_t context_size = 30;
    std::size_t strategy_hidden_size = 128;
    std::size_t sequence_capacity = 16;
    std::size_t character_embedding_size = 16;
    std::size_t opponent_embedding_size = 16;
    std::size_t opponent_count = 28;
    std::size_t replay_capacity = 20000;
    std::size_t minimum_replay_size = 256;
    std::size_t batch_size = 64;
    std::size_t training_updates_per_transition = 1;
    std::uint64_t target_update_interval = 500;
    std::uint64_t initial_training_updates = 0;
    // Flattened [character][opponent] victory counts loaded from the shared
    // metadata checkpoint. They drive replay sampling without changing the
    // recurrent checkpoint architecture.
    std::vector<std::uint64_t> initial_matchup_wins;
    float gamma = 0.995f;
    float q_value_limit = 25.0f;
    float terminal_replay_fraction = 0.25f;
    float hard_negative_replay_fraction = 0.25f;
    float winning_replay_fraction = 0.30f;
    float winning_replay_min_fraction = 0.10f;
    std::size_t winning_replay_ramp_episodes = 5;
    std::size_t winning_replay_capacity = 4096;
    float hard_negative_q_threshold = 1.0f;
    std::size_t n_step_options = 3;
    float learning_rate = 1.0e-4f;
    float resnet_learning_rate = 1.0e-6f;
    float gradient_clip = 5.0f;
    std::uint32_t random_seed = 0x524c444eU;
    int device_id = 0;
    bool training_enabled = true;
    bool checkpoint_writes_enabled = true;
    bool resume = true;
    bool conservative_bootstrap = true;
    bool trainable_resnet = false;
    std::size_t resnet_batch_size = 4;
    std::size_t resnet_update_interval = 16;
    std::size_t resnet_replay_images = 4096;
};

struct RecurrentPolicyBackendStats {
    std::uint64_t training_updates = 0;
    std::size_t replay_size = 0;
    std::size_t winning_replay_size = 0;
    std::uint64_t winning_episodes = 0;
    std::size_t matchup_count = 0;
    std::size_t sequence_size = 0;
    float last_loss = 0.0f;
    float last_predicted_q_mean = 0.0f;
    float last_td_target_mean = 0.0f;
    float last_bootstrap_mean = 0.0f;
    float last_q_target_gap = 0.0f;
    float last_abs_td_error = 0.0f;
    float last_q_target_clipped_fraction = 0.0f;
    float last_terminal_sample_fraction = 0.0f;
    float last_terminal_predicted_q_mean = 0.0f;
    float last_terminal_td_target_mean = 0.0f;
    float last_critic_disagreement = 0.0f;
    float last_hard_negative_sample_fraction = 0.0f;
    float last_winning_sample_fraction = 0.0f;
    float effective_winning_replay_fraction = 0.0f;
    std::uint64_t current_matchup_winning_episodes = 0;
    double parameter_checksum = 0.0;
    double resnet_parameter_checksum = 0.0;
    std::uint64_t resnet_training_updates = 0;
    bool checkpoint_loaded = false;
    bool checkpoint_migrated = false;
    bool resnet_checkpoint_loaded = false;
    bool trainable_resnet = false;
    std::string backend = "cpp";
    std::string device;
};

struct RecurrentPolicyCloseResult {
    RecurrentPolicyBackendStats stats;
    bool training_performed = false;
};

// Recurrent policy facade.  The native backend owns the trainable LSTM, twin
// DQN, replay and BaseAI AdamW state in-process.  The former Python sidecar is
// retained behind GAMEAGENT_RL_BACKEND=python for parity/fallback only.
class RecurrentPolicyBackend {
public:
    explicit RecurrentPolicyBackend(RecurrentPolicyBackendConfig config);
    ~RecurrentPolicyBackend();

    RecurrentPolicyBackend(const RecurrentPolicyBackend&) = delete;
    RecurrentPolicyBackend& operator=(const RecurrentPolicyBackend&) = delete;
    RecurrentPolicyBackend(RecurrentPolicyBackend&&) = delete;
    RecurrentPolicyBackend& operator=(RecurrentPolicyBackend&&) = delete;

    void observe(
        std::span<const float> visual,
        std::span<const std::uint8_t> jpeg,
        std::uint64_t frame_id,
        std::span<const float> controller,
        std::span<const float> context,
        bool episode_started
    );
    std::vector<float> q_values(
        std::size_t character,
        std::size_t opponent,
        std::size_t action_count
    );
    void begin_transition(
        std::size_t action,
        std::size_t character,
        std::size_t opponent,
        float selected_q
    );
    RecurrentPolicyCloseResult close_transition(
        std::size_t duration,
        float reward,
        bool terminal,
        bool successful_episode,
        std::span<const std::uint8_t> next_action_mask
    );
    // Finalizes a round that ended between options. The terminal reward is
    // attached to the most recent pending option before its n-step suffix is
    // flushed, and a successful trajectory is promoted to the win reserve.
    RecurrentPolicyCloseResult finalize_episode(
        float terminal_reward,
        bool successful_episode,
        std::span<const std::uint8_t> next_action_mask
    );
    void reset();
    void save(
        const std::filesystem::path& path = {},
        bool include_replay = false
    );

    [[nodiscard]] const RecurrentPolicyBackendStats& stats() const;
    [[nodiscard]] const RecurrentPolicyBackendConfig& config() const;

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace gameagent
