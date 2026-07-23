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
    std::size_t max_jpeg_bytes = 8U * 1024U * 1024U;
    std::chrono::milliseconds decision_interval{50};
    std::chrono::milliseconds failsafe_timeout{200};
    std::optional<std::string> api_token;
    std::optional<std::uint32_t> random_seed;
    std::filesystem::path metrics_path = "metrics.jsonl";
    std::filesystem::path capture_directory = "captures";
    bool save_frames = false;
    std::size_t save_every_n_frames = 1;
    std::size_t max_saved_frames = 300;
};

class WebSocketServer {
public:
    explicit WebSocketServer(WebSocketServerOptions options);

    int run(const std::atomic_bool& stop_requested);

private:
    WebSocketServerOptions options_;
};

}  // namespace gameagent
