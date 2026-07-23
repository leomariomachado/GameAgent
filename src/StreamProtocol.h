#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace gameagent {

inline constexpr std::string_view kFrameEnvelopeMagic = "GAF1";
inline constexpr std::size_t kFrameEnvelopeHeaderBytes = 24;
inline constexpr std::uint32_t kActionProtocolVersion = 2;
inline constexpr std::string_view kActionMode = "state";

struct FrameEnvelope {
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_ms = 0;
    std::vector<std::uint8_t> jpeg;
};

struct ActionAcknowledgement {
    std::uint64_t decision_id = 0;
    std::int64_t applied_timestamp_ms = 0;
    std::vector<int> applied_commands;
};

using ClientTextMessage = std::variant<FrameEnvelope, ActionAcknowledgement>;

FrameEnvelope parse_binary_frame_envelope(std::span<const std::uint8_t> payload);

std::vector<std::uint8_t> make_binary_frame_envelope(
    std::uint64_t frame_id,
    std::int64_t timestamp_ms,
    std::span<const std::uint8_t> jpeg
);

std::vector<std::uint8_t> decode_base64(std::string_view encoded);

ClientTextMessage parse_client_text_message(std::string_view payload);

std::string websocket_accept_key(std::string_view client_key);

std::vector<std::uint8_t> make_websocket_server_frame(
    std::uint8_t opcode,
    std::span<const std::uint8_t> payload
);

}  // namespace gameagent
