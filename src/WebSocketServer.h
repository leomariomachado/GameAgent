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
    std::chrono::milliseconds decision_interval{50};
    std::chrono::milliseconds failsafe_timeout{200};
    std::optional<std::string> api_token;
    std::optional<std::uint32_t> random_seed;
    bool mock_facing_right = true;
    bool temporal_model_enabled = false;
    std::filesystem::path resnet50_checkpoint =
        "/workspace/resnet50_best.ckpt";
    std::filesystem::path temporal_policy_checkpoint;
    std::size_t temporal_visual_projection_size = 256;
    std::size_t temporal_hidden_size = 256;
    int temporal_device_id = 0;
    std::filesystem::path metrics_path = "metrics.jsonl";
    std::filesystem::path capture_directory = "captures";
    std::filesystem::path training_capture_directory =
        "/workspace/gameagent/training_captures";
    bool save_frames = false;
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
