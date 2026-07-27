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

        const auto acknowledgement = gameagent::parse_client_text_message(
            R"({"protocol_version":2,"type":"action_ack","decision_id":7,"applied_commands":[],"applied_timestamp_ms":1784740000200})"
        );
        require(
            std::holds_alternative<gameagent::ActionAcknowledgement>(acknowledgement),
            "action_ack was not recognized"
        );
        const auto& parsed_ack =
            std::get<gameagent::ActionAcknowledgement>(acknowledgement);
        require(parsed_ack.decision_id == 7, "action_ack decision ID mismatch");
        require(parsed_ack.applied_commands.empty(),
                "empty applied command state was not preserved");

        const auto reset_ack = gameagent::parse_client_text_message(
            R"({"protocol_version":2,"type":"training_reset_ack","reset_id":3,"success":true,"completed_timestamp_ms":1784740000300,"message":"save state loaded"})"
        );
        require(
            std::holds_alternative<
                gameagent::TrainingResetAcknowledgement
            >(reset_ack),
            "training_reset_ack was not recognized"
        );
        const auto& parsed_reset_ack =
            std::get<gameagent::TrainingResetAcknowledgement>(reset_ack);
        require(
            parsed_reset_ack.reset_id == 3 &&
                parsed_reset_ack.success &&
                parsed_reset_ack.message == "save state loaded",
            "training_reset_ack fields were not preserved"
        );

        const auto session = gameagent::parse_client_text_message(
            R"({"protocol_version":2,"type":"session_config","player_character":"scorpion","opponent_character":"subzero","player_side":"left","initial_facing":"right"})"
        );
        require(
            std::holds_alternative<gameagent::SessionConfiguration>(session),
            "session_config was not recognized"
        );
        const auto& parsed_session =
            std::get<gameagent::SessionConfiguration>(session);
        require(
            parsed_session.player_character == "scorpion" &&
                parsed_session.opponent_character == "subzero" &&
                parsed_session.player_side == "left" &&
                parsed_session.initial_facing == "right",
            "session_config fields were not preserved"
        );

        std::cout << "[PASS] GameAgent stream protocol test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
