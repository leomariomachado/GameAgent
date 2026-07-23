#pragma once

#include <json.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace gameagent {

inline constexpr std::size_t kExpectedFrameCount = 30;

struct ProtocolResponse {
    int http_status = 500;
    nlohmann::json body;
    std::filesystem::path capture_directory;
};

class MockProtocol {
public:
    explicit MockProtocol(
        std::filesystem::path capture_root,
        std::optional<std::uint32_t> random_seed = std::nullopt
    );

    ProtocolResponse process(std::string_view request_body);

    [[nodiscard]] const std::filesystem::path& capture_root() const;

private:
    std::vector<int> make_random_commands();
    std::string make_request_id();

    std::filesystem::path capture_root_;
    std::mt19937 random_engine_;
    std::mutex random_mutex_;
    std::atomic<std::uint64_t> request_counter_{0};
};

}  // namespace gameagent
