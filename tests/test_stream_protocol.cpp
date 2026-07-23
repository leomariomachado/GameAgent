#include "StreamProtocol.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    try {
        require(
            gameagent::websocket_accept_key("dGhlIHNhbXBsZSBub25jZQ==") ==
                "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=",
            "RFC 6455 handshake accept key mismatch"
        );

        const std::array<std::uint8_t, 6> jpeg{0xffU, 0xd8U, 1U, 2U, 0xffU, 0xd9U};
        const auto encoded = gameagent::make_binary_frame_envelope(
            42U, 1'784'740'000'123LL, jpeg
        );
        const auto decoded = gameagent::parse_binary_frame_envelope(encoded);
        require(decoded.frame_id == 42U, "frame_id was not preserved");
        require(decoded.timestamp_ms == 1'784'740'000'123LL, "timestamp was not preserved");
        require(decoded.jpeg == std::vector<std::uint8_t>(jpeg.begin(), jpeg.end()),
                "JPEG payload was not preserved");

        const std::string action = R"({"commands":[9,3]})";
        const auto frame = gameagent::make_websocket_server_frame(
            0x1U,
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(action.data()), action.size()
            )
        );
        require(frame.size() == action.size() + 2U, "short WebSocket frame has wrong size");
        require(frame[0] == 0x81U, "server text frame must be final text");
        require(frame[1] == action.size(), "short WebSocket payload length mismatch");

        std::cout << "[PASS] GameAgent stream protocol test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
