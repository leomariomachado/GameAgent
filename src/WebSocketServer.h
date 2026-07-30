#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace gameagent {

struct WebSocketServerOptions {
    std::string host = "0.0.0.0";
    std::uint16_t port = 8080;
    std::size_t frame_buffer_capacity = 32;
    std::size_t observation_frames = 8;
    std::size_t applied_action_history_size = 8;
    std::size_t max_jpeg_bytes = 8U * 1024U * 1024U;
    std::size_t persistence_queue_capacity = 256;
    std::chrono::milliseconds decision_interval{50};
    std::chrono::milliseconds failsafe_timeout{200};
    std::chrono::milliseconds reinforcement_reward_log_interval{5000};
    std::optional<std::string> api_token;
    std::optional<std::uint32_t> random_seed;
    bool default_facing_right = true;
    bool temporal_model_enabled = false;
    bool reinforcement_learning_enabled = false;
    bool reinforcement_training_enabled = true;
    bool reinforcement_resume = true;
    bool reinforcement_trainable_lstm = true;
    bool reinforcement_trainable_resnet = false;
    bool training_reset_state_enabled = true;
    bool player_is_left = true;
    std::filesystem::path resnet50_checkpoint =
        "/workspace/resnet50_best.ckpt";
    std::filesystem::path temporal_policy_checkpoint;
    std::size_t temporal_visual_projection_size = 256;
    std::size_t temporal_hidden_size = 256;
    int temporal_device_id = 0;
    std::filesystem::path reinforcement_checkpoint_directory =
        "/workspace/gameagent/rl_checkpoints";
    std::filesystem::path recurrent_service_script =
        "/tmp/GameAgent/scripts/recurrent_dqn_service.py";
    std::size_t reinforcement_network_hidden_size = 128;
    std::size_t reinforcement_replay_capacity = 20000;
    std::size_t reinforcement_minimum_replay_size = 256;
    std::size_t reinforcement_batch_size = 64;
    std::uint64_t reinforcement_target_update_interval = 500;
    std::uint64_t reinforcement_epsilon_decay_selections = 100000;
    float reinforcement_gamma = 0.99f;
    float reinforcement_learning_rate = 1.0e-4f;
    float reinforcement_resnet_learning_rate = 1.0e-6f;
    std::size_t reinforcement_resnet_batch_size = 4;
    std::size_t reinforcement_resnet_update_interval = 16;
    std::size_t reinforcement_resnet_replay_images = 4096;
    float reinforcement_epsilon_start = 1.0f;
    float reinforcement_epsilon_final = 0.05f;
    float reinforcement_scripted_exploration_probability = 0.50f;
    std::filesystem::path metrics_path = "metrics.jsonl";
    std::filesystem::path capture_directory = "captures";
    std::filesystem::path training_capture_directory =
        "/workspace/gameagent/training_captures";
    bool save_frames = false;
    std::chrono::milliseconds debug_capture_interval{3000};
    bool collect_training_data = false;
};

class WebSocketServer {
public:
    explicit WebSocketServer(WebSocketServerOptions options);

    int run(const std::atomic_bool& stop_requested);

private:
    WebSocketServerOptions options_;
};

}  // namespace gameagent
