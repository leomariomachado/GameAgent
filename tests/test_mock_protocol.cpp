#include "MockProtocol.h"

#include <json.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include <unistd.h>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

nlohmann::json make_request(std::size_t frame_count) {
    nlohmann::json frames = nlohmann::json::array();
    for (std::size_t index = 0; index < frame_count; ++index) {
        frames.push_back({
            {"timestamp_ms", 1784740000000LL + static_cast<long long>(index * 100)},
            // Minimal JPEG-like byte sequence: FF D8 FF D9.
            {"jpeg_base64", "/9j/2Q=="},
        });
    }
    return nlohmann::json{
        {"protocol_version", 1},
        {"emulator", "zsnes"},
        {"window_seconds", 3},
        {"sample_fps", 10},
        {"frames", std::move(frames)},
    };
}

}  // namespace

int main() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("gameagent_mock_test_" + std::to_string(getpid()) + '_' + std::to_string(unique));

    try {
        gameagent::MockProtocol protocol(root, 12345U);

        const gameagent::ProtocolResponse success = protocol.process(
            make_request(gameagent::kExpectedFrameCount).dump()
        );
        require(success.http_status == 200, "valid request should return HTTP 200");
        require(success.body.is_object(), "response should be a JSON object");
        require(success.body.size() == 1, "response should contain only commands");
        require(success.body.contains("commands"), "response should contain commands");
        require(success.body["commands"].is_array(), "commands should be an array");

        const std::size_t command_count = success.body["commands"].size();
        require(command_count >= 3 && command_count <= 10, "command count should be 3..10");

        std::set<int> unique_commands;
        for (const auto& command : success.body["commands"]) {
            require(command.is_number_integer(), "each command should be an integer");
            const int value = command.get<int>();
            require(value >= 0 && value <= 9, "each command should be in the 0..9 range");
            unique_commands.insert(value);
        }
        require(
            unique_commands.size() == command_count,
            "simultaneous commands should not contain duplicates"
        );

        require(
            std::filesystem::exists(success.capture_directory / "metadata.json"),
            "metadata.json should be saved"
        );
        require(
            std::filesystem::exists(success.capture_directory / "response.json"),
            "response.json should be saved"
        );

        std::size_t saved_jpegs = 0;
        for (const auto& entry : std::filesystem::directory_iterator(success.capture_directory)) {
            if (entry.path().extension() == ".jpg") {
                ++saved_jpegs;
            }
        }
        require(saved_jpegs == 30, "exactly 30 JPEG files should be saved");

        const gameagent::ProtocolResponse short_window = protocol.process(make_request(29).dump());
        require(short_window.http_status == 422, "29 frames should be rejected");
        require(
            short_window.body.value("expected_frames", 0) == 30,
            "frame-count error should report 30 expected frames"
        );

        const gameagent::ProtocolResponse malformed = protocol.process("not-json");
        require(malformed.http_status == 400, "malformed JSON should be rejected");

        std::filesystem::remove_all(root);
        std::cout << "[PASS] GameAgent mock protocol test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
