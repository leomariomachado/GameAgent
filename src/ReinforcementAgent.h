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
    float gamma = 0.99f;
    float learning_rate = 1.0e-4f;
    float adam_beta1 = 0.9f;
    float adam_beta2 = 0.999f;
    float adam_epsilon = 1.0e-8f;
    float gradient_clip = 5.0f;
    float epsilon_start = 1.0f;
    float epsilon_final = 0.05f;
    std::uint64_t epsilon_decay_selections = 100000;
    float scripted_exploration_probability = 0.50f;
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
    // Enables the CUDA/PyTorch recurrent Double-DQN backend. The legacy
    // feed-forward DQN remains available for unit tests and checkpoint
    // migration, but production RL uses a trainable LSTM by default.
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
    float episode_return = 0.0f;
    float training_loss = 0.0f;
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
    float current_episode_return = 0.0f;
    float best_episode_return = 0.0f;
    double shared_parameter_checksum = 0.0;
    std::size_t shared_replay_matchups = 0;
    bool checkpoint_loaded = false;
    bool trainable_lstm = false;
    bool recurrent_checkpoint_loaded = false;
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
