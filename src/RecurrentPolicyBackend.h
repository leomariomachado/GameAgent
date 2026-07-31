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
    std::filesystem::path service_script;
    std::filesystem::path checkpoint_path;
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
    float gamma = 0.99f;
    float learning_rate = 1.0e-4f;
    float resnet_learning_rate = 1.0e-6f;
    float gradient_clip = 5.0f;
    std::uint32_t random_seed = 0x524c444eU;
    int device_id = 0;
    bool training_enabled = true;
    bool checkpoint_writes_enabled = true;
    bool resume = true;
    bool trainable_resnet = false;
    std::size_t resnet_batch_size = 4;
    std::size_t resnet_update_interval = 16;
    std::size_t resnet_replay_images = 4096;
};

struct RecurrentPolicyBackendStats {
    std::uint64_t training_updates = 0;
    std::size_t replay_size = 0;
    std::size_t matchup_count = 0;
    std::size_t sequence_size = 0;
    float last_loss = 0.0f;
    double parameter_checksum = 0.0;
    double resnet_parameter_checksum = 0.0;
    std::uint64_t resnet_training_updates = 0;
    bool checkpoint_loaded = false;
    bool resnet_checkpoint_loaded = false;
    bool trainable_resnet = false;
    std::string device;
};

struct RecurrentPolicyCloseResult {
    RecurrentPolicyBackendStats stats;
    bool training_performed = false;
};

// Bidirectional local sidecar connection. PyTorch owns the trainable recurrent
// graph and optimizer on CUDA; C++ remains responsible for deterministic
// controller option timing and the public WebSocket protocol.
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
        std::size_t opponent
    );
    RecurrentPolicyCloseResult close_transition(
        std::size_t duration,
        float reward,
        bool terminal
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
