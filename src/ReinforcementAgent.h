#pragma once

#include "MockAgent.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gameagent {

struct ReinforcementAgentConfig {
    std::size_t input_size = 286;
    std::size_t network_hidden_size = 128;
    std::size_t replay_capacity = 20000;
    std::size_t minimum_replay_size = 256;
    std::size_t batch_size = 64;
    std::size_t training_updates_per_transition = 1;
    std::uint64_t target_update_interval = 500;
    float gamma = 0.995f;
    float q_value_limit = 25.0f;
    float terminal_replay_fraction = 0.25f;
    float hard_negative_replay_fraction = 0.25f;
    float hard_negative_q_threshold = 1.0f;
    std::size_t n_step_options = 3;
    bool conservative_bootstrap = true;
    float learning_rate = 1.0e-4f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_epsilon = 1.0e-8f;
    float gradient_clip = 5.0f;
    float epsilon_start = 1.0f;
    float epsilon_final = 0.05f;
    std::uint64_t epsilon_decay_selections = 100000;
    float scripted_exploration_probability = 0.50f;
    // Neutral remains in the action head for checkpoint compatibility, but is
    // masked during a spatially valid fight. The penalty still protects
    // uncertain/fallback observations from learning a no-op shortcut.
    float neutral_option_penalty = 0.20f;
    bool suppress_neutral_during_fight = true;
    float combo_max_distance = 0.22f;
    float close_attack_max_distance = 0.24f;
    float projectile_preferred_min_distance = 0.28f;
    float projectile_q_preference_margin = 1.0f;
    float disable_cloud_min_distance = 0.12f;
    float disable_cloud_max_distance = 0.55f;
    float advance_min_distance = 0.18f;
    float defense_max_distance = 0.32f;
    bool dynamic_facing = true;
    std::size_t disable_cloud_cooldown_options = 2;
    // Damage arrives at different delays for melee, teleport and projectile
    // options. These post-program settle windows keep the option responsible
    // for the corresponding health change.
    std::size_t melee_impact_credit_steps = 2;
    std::size_t teleport_impact_credit_steps = 4;
    std::size_t projectile_impact_credit_min_steps = 4;
    std::size_t projectile_impact_credit_max_steps = 8;
    // Every Nth training episode runs nearly greedily while still writing its
    // failures to replay. This closes the train/evaluation distribution gap.
    std::size_t greedy_training_episode_interval = 10;
    float greedy_training_epsilon = 0.05f;
    std::uint32_t random_seed = 0x524c444eU;
    FighterCharacter character = FighterCharacter::subzero;
    OpponentCharacter opponent_character = OpponentCharacter::unknown;
    FighterFacing facing = FighterFacing::right;
    std::size_t character_embedding_size = 16;
    std::size_t opponent_embedding_size = 16;
    bool training_enabled = true;
    bool checkpoint_writes_enabled = true;
    // Production servers can checkpoint less often to keep large recurrent
    // snapshots off the latency-sensitive decision path. A value of 1 keeps
    // the original save-on-every-episode behavior.
    std::size_t checkpoint_interval_episodes = 1;
    bool resume = true;
    bool save_replay = true;
    // Native C++/CUDA is the production default. Python remains a reference
    // backend for checkpoint conversion and controlled parity comparisons.
    std::string recurrent_backend = "cpp";
    bool trainable_lstm = false;
    bool trainable_resnet = false;
    std::size_t temporal_visual_size = 2048;
    std::size_t temporal_projection_size = 256;
    std::size_t temporal_controller_size = 12;
    std::size_t temporal_hidden_size = 256;
    std::size_t temporal_sequence_capacity = 16;
    int temporal_device_id = 0;
    std::filesystem::path resnet50_checkpoint_path;
    float resnet_learning_rate = 1.0e-6f;
    std::size_t resnet_batch_size = 4;
    std::size_t resnet_update_interval = 16;
    std::size_t resnet_replay_images = 4096;
    std::filesystem::path recurrent_service_script;
    std::filesystem::path recurrent_checkpoint_path;
    std::filesystem::path recurrent_migration_checkpoint_path;
    std::filesystem::path checkpoint_path;
    std::filesystem::path best_checkpoint_path;
};

struct ReinforcementStepInput {
    std::span<const float> state;
    std::span<const float> temporal_visual_feature;
    std::span<const std::uint8_t> temporal_jpeg;
    std::span<const float> temporal_controller_features;
    float reward = 0.0f;
    bool action_allowed = false;
    bool episode_started = false;
    bool episode_ended = false;
    bool terminal = false;
    std::uint64_t episode_id = 0;
    std::uint64_t frame_id = 0;
    std::uint64_t decision_id = 0;
    float fighter_distance = 0.40f;
    float player_x = 0.25f;
    float opponent_x = 0.75f;
    bool spatial_valid = false;
    bool opponent_attacking = false;
};

struct ReinforcementDecision {
    std::vector<int> commands;
    std::string option_name = "release";
    std::string selection_source = "fight_gate";
    std::size_t option_index = 0;
    std::size_t option_step = 0;
    std::size_t option_steps = 0;
    float epsilon = 0.0f;
    float selected_q = 0.0f;
    float reward_consumed = 0.0f;
    float neutral_penalty = 0.0f;
    std::size_t allowed_options = 0;
    std::size_t impact_credit_step = 0;
    std::size_t impact_credit_steps = 0;
    std::size_t disable_cloud_cooldown_remaining = 0;
    bool commands_mirrored = false;
    bool greedy_training_episode = false;
    float episode_return = 0.0f;
    float training_loss = 0.0f;
    float predicted_q_mean = 0.0f;
    float td_target_mean = 0.0f;
    float bootstrap_mean = 0.0f;
    float q_target_gap = 0.0f;
    float abs_td_error = 0.0f;
    float q_target_clipped_fraction = 0.0f;
    float terminal_sample_fraction = 0.0f;
    float terminal_predicted_q_mean = 0.0f;
    float terminal_td_target_mean = 0.0f;
    float critic_disagreement = 0.0f;
    float hard_negative_sample_fraction = 0.0f;
    std::size_t replay_size = 0;
    std::uint64_t training_updates = 0;
    bool training_performed = false;
    bool checkpoint_saved = false;
};

struct ReinforcementAgentStats {
    std::uint64_t action_selections = 0;
    std::uint64_t training_updates = 0;
    std::uint64_t episodes_completed = 0;
    std::size_t replay_size = 0;
    float epsilon = 0.0f;
    float last_training_loss = 0.0f;
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
    float current_episode_return = 0.0f;
    float best_episode_return = 0.0f;
    double shared_parameter_checksum = 0.0;
    std::size_t shared_replay_matchups = 0;
    bool checkpoint_loaded = false;
    bool trainable_lstm = false;
    bool recurrent_checkpoint_loaded = false;
    bool recurrent_checkpoint_migrated = false;
    bool trainable_resnet = false;
    bool resnet_checkpoint_loaded = false;
    std::uint64_t resnet_training_updates = 0;
    double resnet_parameter_checksum = 0.0;
    std::string training_device = "cpu";
};

// Online semi-Markov Double DQN with a shared strategy trunk, learned
// player/opponent embeddings, one action head per controlled character and a
// matchup-balanced replay. A transition is closed at the end of an option, so
// delayed damage is credited to the whole move rather than an arbitrary
// controller state.
class ReinforcementAgent {
public:
    explicit ReinforcementAgent(ReinforcementAgentConfig config);
    ~ReinforcementAgent();

    ReinforcementAgent(const ReinforcementAgent&) = delete;
    ReinforcementAgent& operator=(const ReinforcementAgent&) = delete;
    ReinforcementAgent(ReinforcementAgent&&) noexcept;
    ReinforcementAgent& operator=(ReinforcementAgent&&) noexcept;

    ReinforcementDecision step(const ReinforcementStepInput& input);
    // Cancels the in-flight option without creating a replay transition or
    // changing episode return/epsilon. Used when the visual stream freezes.
    void suspend();
    void reset_episode(bool save_checkpoint);
    void save_checkpoint(const std::filesystem::path& path = {});
    void load_checkpoint(const std::filesystem::path& path);

    [[nodiscard]] ReinforcementAgentStats stats() const;
    [[nodiscard]] std::vector<std::string> option_names() const;
    [[nodiscard]] const ReinforcementAgentConfig& config() const;

private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace gameagent
