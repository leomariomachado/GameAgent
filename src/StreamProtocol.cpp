#include "StreamProtocol.h"

#include <json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gameagent {
namespace {

MessageMetadata parse_message_metadata(
    const nlohmann::json& request,
    std::string_view type
) {
    if (!request.contains("protocol_version") ||
        !request["protocol_version"].is_number_integer() ||
        request["protocol_version"].get<int>() !=
            static_cast<int>(kServerProtocolVersion)) {
        throw std::invalid_argument(
            std::string(type) + " protocol_version must be 2"
        );
    }
    if (!request.contains("request_id") ||
        !request["request_id"].is_number_unsigned() ||
        request["request_id"].get<std::uint64_t>() == 0) {
        throw std::invalid_argument(
            std::string(type) + " request_id must be a positive uint64"
        );
    }
    if (!request.contains("timestamp_ms") ||
        !request["timestamp_ms"].is_number_integer() ||
        request["timestamp_ms"].get<std::int64_t>() <= 0) {
        throw std::invalid_argument(
            std::string(type) + " timestamp_ms must be positive"
        );
    }
    return {
        .request_id = request["request_id"].get<std::uint64_t>(),
        .timestamp_ms = request["timestamp_ms"].get<std::int64_t>(),
    };
}

std::uint64_t positive_u64(
    const nlohmann::json& request,
    std::string_view field,
    std::string_view type
) {
    const std::string key(field);
    if (!request.contains(key) || !request[key].is_number_unsigned() ||
        request[key].get<std::uint64_t>() == 0) {
        throw std::invalid_argument(
            std::string(type) + " " + key + " must be a positive uint64"
        );
    }
    return request[key].get<std::uint64_t>();
}

std::string required_string(
    const nlohmann::json& request,
    std::string_view field,
    std::string_view type,
    std::size_t maximum = 64U
) {
    const std::string key(field);
    if (!request.contains(key) || !request[key].is_string()) {
        throw std::invalid_argument(
            std::string(type) + " " + key + " must be a string"
        );
    }
    std::string value = request[key].get<std::string>();
    if (value.empty() || value.size() > maximum) {
        throw std::invalid_argument(
            std::string(type) + " " + key + " has invalid length"
        );
    }
    return value;
}

EncounterConfiguration parse_encounter_configuration(
    const nlohmann::json& request,
    std::string_view type
) {
    const nlohmann::json& source =
        request.contains("encounter") ? request["encounter"] : request;
    if (!source.is_object()) {
        throw std::invalid_argument("encounter must be an object");
    }
    const auto parsed_type = parse_encounter_type(
        required_string(source, "encounter_type", type)
    );
    if (!parsed_type.has_value()) {
        throw std::invalid_argument("unsupported encounter_type");
    }
    if (!source.contains("opponents") ||
        !source["opponents"].is_array()) {
        throw std::invalid_argument("encounter opponents must be an array");
    }
    const auto& opponents = source["opponents"];
    if (opponents.empty() ||
        opponents.size() > kMaximumEncounterOpponents) {
        throw std::invalid_argument("encounter requires 1..4 opponents");
    }
    EncounterConfiguration configuration;
    configuration.type = *parsed_type;
    configuration.opponent_count = static_cast<std::uint8_t>(
        source.value<std::size_t>("opponent_count", opponents.size())
    );
    configuration.current_stage = static_cast<std::uint8_t>(
        source.value<std::size_t>("current_stage", 0U)
    );
    if (configuration.opponent_count != opponents.size()) {
        throw std::invalid_argument(
            "opponent_count must match opponents array size"
        );
    }
    for (std::size_t index = 0; index < opponents.size(); ++index) {
        if (!opponents[index].is_string()) {
            throw std::invalid_argument("encounter opponents must be strings");
        }
        const auto opponent = parse_opponent_character(
            opponents[index].get<std::string>()
        );
        if (!opponent.has_value() ||
            *opponent == OpponentCharacter::unknown) {
            throw std::invalid_argument("unsupported encounter opponent");
        }
        configuration.opponents[index] = *opponent;
    }
    configuration.validate();
    if (configuration.current_stage != 0U) {
        throw std::invalid_argument(
            "configure_session encounter current_stage must be 0"
        );
    }
    return configuration;
}

std::uint32_t read_u32_be(std::span<const std::uint8_t> bytes) {
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
        (static_cast<std::uint32_t>(bytes[1]) << 16U) |
        (static_cast<std::uint32_t>(bytes[2]) << 8U) |
        static_cast<std::uint32_t>(bytes[3]);
}

std::uint64_t read_u64_be(std::span<const std::uint8_t> bytes) {
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        value = (value << 8U) | bytes[index];
    }
    return value;
}

void append_u32_be(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
        output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

void append_u64_be(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) {
        output.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

std::array<std::uint8_t, 20> sha1(std::string_view input) {
    std::vector<std::uint8_t> padded(input.begin(), input.end());
    const std::uint64_t bit_length = static_cast<std::uint64_t>(padded.size()) * 8U;
    padded.push_back(0x80U);
    while (padded.size() % 64U != 56U) {
        padded.push_back(0U);
    }
    append_u64_be(padded, bit_length);

    std::uint32_t h0 = 0x67452301U;
    std::uint32_t h1 = 0xefcdab89U;
    std::uint32_t h2 = 0x98badcfeU;
    std::uint32_t h3 = 0x10325476U;
    std::uint32_t h4 = 0xc3d2e1f0U;

    for (std::size_t block = 0; block < padded.size(); block += 64U) {
        std::array<std::uint32_t, 80> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            words[index] = read_u32_be(
                std::span<const std::uint8_t>(padded).subspan(block + index * 4U, 4U)
            );
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            words[index] = std::rotl(
                words[index - 3] ^ words[index - 8] ^ words[index - 14] ^
                    words[index - 16],
                1
            );
        }

        std::uint32_t a = h0;
        std::uint32_t b = h1;
        std::uint32_t c = h2;
        std::uint32_t d = h3;
        std::uint32_t e = h4;

        for (std::size_t index = 0; index < words.size(); ++index) {
            std::uint32_t function = 0;
            std::uint32_t constant = 0;
            if (index < 20) {
                function = (b & c) | ((~b) & d);
                constant = 0x5a827999U;
            } else if (index < 40) {
                function = b ^ c ^ d;
                constant = 0x6ed9eba1U;
            } else if (index < 60) {
                function = (b & c) | (b & d) | (c & d);
                constant = 0x8f1bbcdcU;
            } else {
                function = b ^ c ^ d;
                constant = 0xca62c1d6U;
            }
            const std::uint32_t temporary = std::rotl(a, 5) + function + e +
                constant + words[index];
            e = d;
            d = c;
            c = std::rotl(b, 30);
            b = a;
            a = temporary;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    std::array<std::uint8_t, 20> digest{};
    const std::array<std::uint32_t, 5> hashes{h0, h1, h2, h3, h4};
    for (std::size_t index = 0; index < hashes.size(); ++index) {
        const std::uint32_t value = hashes[index];
        digest[index * 4U] = static_cast<std::uint8_t>((value >> 24U) & 0xffU);
        digest[index * 4U + 1U] = static_cast<std::uint8_t>((value >> 16U) & 0xffU);
        digest[index * 4U + 2U] = static_cast<std::uint8_t>((value >> 8U) & 0xffU);
        digest[index * 4U + 3U] = static_cast<std::uint8_t>(value & 0xffU);
    }
    return digest;
}

std::string encode_base64_impl(std::span<const std::uint8_t> bytes) {
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve(((bytes.size() + 2U) / 3U) * 4U);

    for (std::size_t offset = 0; offset < bytes.size(); offset += 3U) {
        const std::size_t remaining = bytes.size() - offset;
        const std::uint32_t block =
            (static_cast<std::uint32_t>(bytes[offset]) << 16U) |
            (remaining > 1U ? static_cast<std::uint32_t>(bytes[offset + 1U]) << 8U : 0U) |
            (remaining > 2U ? static_cast<std::uint32_t>(bytes[offset + 2U]) : 0U);
        encoded.push_back(alphabet[(block >> 18U) & 0x3fU]);
        encoded.push_back(alphabet[(block >> 12U) & 0x3fU]);
        encoded.push_back(remaining > 1U ? alphabet[(block >> 6U) & 0x3fU] : '=');
        encoded.push_back(remaining > 2U ? alphabet[block & 0x3fU] : '=');
    }
    return encoded;
}

}  // namespace

std::string encode_base64(std::span<const std::uint8_t> bytes) {
    return encode_base64_impl(bytes);
}

FrameEnvelope parse_binary_frame_envelope(std::span<const std::uint8_t> payload) {
    if (payload.size() < kFrameEnvelopeHeaderBytes) {
        throw std::invalid_argument("binary frame envelope is shorter than 24 bytes");
    }
    if (!std::equal(kFrameEnvelopeMagic.begin(), kFrameEnvelopeMagic.end(), payload.begin())) {
        throw std::invalid_argument("binary frame envelope has invalid magic; expected GAF1");
    }

    const std::uint64_t frame_id = read_u64_be(payload.subspan(4U, 8U));
    const std::int64_t timestamp_ms = static_cast<std::int64_t>(
        read_u64_be(payload.subspan(12U, 8U))
    );
    const std::uint32_t jpeg_size = read_u32_be(payload.subspan(20U, 4U));
    if (jpeg_size == 0U || payload.size() != kFrameEnvelopeHeaderBytes + jpeg_size) {
        throw std::invalid_argument("binary frame envelope JPEG size does not match payload");
    }

    return FrameEnvelope{
        .frame_id = frame_id,
        .timestamp_ms = timestamp_ms,
        .jpeg = std::vector<std::uint8_t>(
            payload.begin() + static_cast<std::ptrdiff_t>(kFrameEnvelopeHeaderBytes),
            payload.end()
        ),
    };
}

std::vector<std::uint8_t> make_binary_frame_envelope(
    std::uint64_t frame_id,
    std::int64_t timestamp_ms,
    std::span<const std::uint8_t> jpeg
) {
    if (jpeg.empty() || jpeg.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("JPEG must contain 1 to UINT32_MAX bytes");
    }
    std::vector<std::uint8_t> output;
    output.reserve(kFrameEnvelopeHeaderBytes + jpeg.size());
    output.insert(output.end(), kFrameEnvelopeMagic.begin(), kFrameEnvelopeMagic.end());
    append_u64_be(output, frame_id);
    append_u64_be(output, static_cast<std::uint64_t>(timestamp_ms));
    append_u32_be(output, static_cast<std::uint32_t>(jpeg.size()));
    output.insert(output.end(), jpeg.begin(), jpeg.end());
    return output;
}

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (encoded.starts_with("data:")) {
        const std::size_t comma = encoded.find(',');
        if (comma == std::string_view::npos) {
            throw std::invalid_argument("invalid Base64 data URL");
        }
        encoded.remove_prefix(comma + 1U);
    }

    std::string compact;
    compact.reserve(encoded.size());
    for (const unsigned char character : encoded) {
        if (!std::isspace(character)) {
            compact.push_back(static_cast<char>(character));
        }
    }
    if (compact.empty() || compact.size() % 4U != 0U) {
        throw std::invalid_argument("invalid Base64 length");
    }

    const auto value_of = [&](char character) -> std::uint32_t {
        const std::size_t position = alphabet.find(character);
        if (position == std::string_view::npos) {
            throw std::invalid_argument("invalid Base64 character");
        }
        return static_cast<std::uint32_t>(position);
    };

    std::vector<std::uint8_t> decoded;
    decoded.reserve((compact.size() / 4U) * 3U);
    for (std::size_t offset = 0; offset < compact.size(); offset += 4U) {
        const bool final_block = offset + 4U == compact.size();
        const bool padding2 = compact[offset + 2U] == '=';
        const bool padding3 = compact[offset + 3U] == '=';
        if (compact[offset] == '=' || compact[offset + 1U] == '=' ||
            (!final_block && (padding2 || padding3)) || (padding2 && !padding3)) {
            throw std::invalid_argument("invalid Base64 padding");
        }
        const std::uint32_t block =
            (value_of(compact[offset]) << 18U) |
            (value_of(compact[offset + 1U]) << 12U) |
            (padding2 ? 0U : value_of(compact[offset + 2U]) << 6U) |
            (padding3 ? 0U : value_of(compact[offset + 3U]));
        decoded.push_back(static_cast<std::uint8_t>((block >> 16U) & 0xffU));
        if (!padding2) {
            decoded.push_back(static_cast<std::uint8_t>((block >> 8U) & 0xffU));
        }
        if (!padding3) {
            decoded.push_back(static_cast<std::uint8_t>(block & 0xffU));
        }
    }
    return decoded;
}

ClientTextMessage parse_client_text_message(std::string_view payload) {
    const nlohmann::json request = nlohmann::json::parse(payload);
    if (!request.is_object()) {
        throw std::invalid_argument("client text message must be a JSON object");
    }

    const std::string type = request.value("type", "frame");
    if (type == "configure_session") {
        const MessageMetadata metadata =
            parse_message_metadata(request, type);
        std::optional<std::string> player_side;
        std::optional<std::string> initial_facing;
        if (request.contains("player_side")) {
            player_side = required_string(request, "player_side", type);
            if (*player_side != "left" && *player_side != "right") {
                throw std::invalid_argument(
                    "configure_session player_side must be left or right"
                );
            }
        }
        if (request.contains("initial_facing")) {
            initial_facing = required_string(
                request, "initial_facing", type
            );
            if (*initial_facing != "left" && *initial_facing != "right") {
                throw std::invalid_argument(
                    "configure_session initial_facing must be left or right"
                );
            }
        }
        return ConfigureSessionRequest{
            .metadata = metadata,
            .player_character = required_string(
                request, "player_character", type, 32U
            ),
            .player_side = std::move(player_side),
            .initial_facing = std::move(initial_facing),
            .encounter = parse_encounter_configuration(request, type),
        };
    }

    if (type == "update_encounter_stage") {
        const MessageMetadata metadata =
            parse_message_metadata(request, type);
        const std::size_t stage = request.value<std::size_t>(
            "current_stage", kMaximumEncounterOpponents
        );
        if (stage >= kMaximumEncounterOpponents) {
            throw std::invalid_argument(
                "update_encounter_stage current_stage must be 0..3"
            );
        }
        const auto opponent = parse_opponent_character(
            required_string(request, "current_opponent", type)
        );
        if (!opponent.has_value() ||
            *opponent == OpponentCharacter::unknown) {
            throw std::invalid_argument(
                "update_encounter_stage current_opponent is unsupported"
            );
        }
        return UpdateEncounterStageRequest{
            .metadata = metadata,
            .episode_id = positive_u64(request, "episode_id", type),
            .stage_transition_id = positive_u64(
                request, "stage_transition_id", type
            ),
            .current_stage = static_cast<std::uint8_t>(stage),
            .current_opponent = *opponent,
        };
    }

    if (type == "encounter_event") {
        const MessageMetadata metadata =
            parse_message_metadata(request, type);
        const auto event = parse_encounter_event(
            required_string(request, "event", type)
        );
        if (!event.has_value() || *event == EncounterEvent::none) {
            throw std::invalid_argument("unsupported encounter event");
        }
        const std::size_t stage = request.value<std::size_t>(
            "encounter_stage", kMaximumEncounterOpponents
        );
        if (stage >= kMaximumEncounterOpponents) {
            throw std::invalid_argument(
                "encounter_event encounter_stage must be 0..3"
            );
        }
        std::uint64_t transition_id = 0;
        if (*event == EncounterEvent::stage_completed ||
            *event == EncounterEvent::stage_started) {
            transition_id = positive_u64(
                request, "stage_transition_id", type
            );
        } else if (request.contains("stage_transition_id")) {
            transition_id = request["stage_transition_id"].get<std::uint64_t>();
        }
        return EncounterEventRequest{
            .metadata = metadata,
            .episode_id = positive_u64(request, "episode_id", type),
            .stage_transition_id = transition_id,
            .event = *event,
            .encounter_stage = static_cast<std::uint8_t>(stage),
        };
    }

    if (type == "set_agent_mode") {
        const MessageMetadata metadata =
            parse_message_metadata(request, type);
        const auto mode = parse_agent_mode(
            required_string(request, "mode", type)
        );
        if (!mode.has_value()) {
            throw std::invalid_argument("unsupported agent mode");
        }
        return SetAgentModeRequest{
            .metadata = metadata,
            .mode = *mode,
            .reset_evaluation_session = request.value(
                "reset_evaluation_session", false
            ),
            .apply_on_next_episode = request.value(
                "apply_on_next_episode", true
            ),
        };
    }

    if (type == "set_exploration") {
        const MessageMetadata metadata =
            parse_message_metadata(request, type);
        const auto mode = parse_exploration_mode(
            required_string(request, "mode", type)
        );
        if (!mode.has_value()) {
            throw std::invalid_argument("unsupported exploration mode");
        }
        const float value = request.value("value", 0.0f);
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::invalid_argument(
                "set_exploration value must be within 0..1"
            );
        }
        return SetExplorationRequest{
            .metadata = metadata,
            .mode = *mode,
            .value = value,
            .reset_matchup_counter = request.value(
                "reset_matchup_counter", false
            ),
        };
    }

    if (type == "request_reset") {
        const MessageMetadata metadata =
            parse_message_metadata(request, type);
        const auto reason = parse_reset_reason(
            required_string(request, "reason", type)
        );
        if (!reason.has_value() || *reason == ResetReason::none) {
            throw std::invalid_argument("unsupported reset reason");
        }
        return RequestReset{
            .metadata = metadata,
            .episode_id = positive_u64(request, "episode_id", type),
            .reason = *reason,
        };
    }

    if (type == "reset_ack") {
        const MessageMetadata metadata =
            parse_message_metadata(request, type);
        if (!request.contains("command_executed") ||
            !request["command_executed"].is_boolean()) {
            throw std::invalid_argument(
                "reset_ack command_executed must be boolean"
            );
        }
        return ResetAck{
            .metadata = metadata,
            .reset_id = positive_u64(request, "reset_id", type),
            .episode_id = positive_u64(request, "episode_id", type),
            .command_executed = request["command_executed"].get<bool>(),
            .executed_timestamp_ms = static_cast<std::int64_t>(
                positive_u64(request, "executed_timestamp_ms", type)
            ),
        };
    }

    if (type == "reset_ready") {
        const MessageMetadata metadata =
            parse_message_metadata(request, type);
        if (!request.contains("stable_initial_state") ||
            !request["stable_initial_state"].is_boolean()) {
            throw std::invalid_argument(
                "reset_ready stable_initial_state must be boolean"
            );
        }
        return ResetReady{
            .metadata = metadata,
            .reset_id = positive_u64(request, "reset_id", type),
            .stable_initial_state =
                request["stable_initial_state"].get<bool>(),
            .first_stable_frame_id = positive_u64(
                request, "first_stable_frame_id", type
            ),
        };
    }

    if (type == "reset_failed") {
        return ResetFailed{
            .metadata = parse_message_metadata(request, type),
            .reset_id = positive_u64(request, "reset_id", type),
            .reason = required_string(request, "reason", type, 512U),
        };
    }
    if (type == "session_config") {
        if (!request.contains("protocol_version") ||
            !request["protocol_version"].is_number_integer() ||
            request["protocol_version"].get<int>() !=
                static_cast<int>(kActionProtocolVersion)) {
            throw std::invalid_argument(
                "session_config protocol_version must be 2"
            );
        }
        if (!request.contains("player_character") ||
            !request["player_character"].is_string()) {
            throw std::invalid_argument(
                "session_config player_character must be a string"
            );
        }
        const std::string player_character =
            request["player_character"].get<std::string>();
        if (player_character.empty() || player_character.size() > 32U) {
            throw std::invalid_argument(
                "session_config player_character must contain 1..32 characters"
            );
        }
        const auto optional_string =
            [&](std::string_view field) -> std::optional<std::string> {
                const std::string key(field);
                if (!request.contains(key)) {
                    return std::nullopt;
                }
                if (!request[key].is_string()) {
                    throw std::invalid_argument(
                        "session_config " + key + " must be a string"
                    );
                }
                std::string value = request[key].get<std::string>();
                if (value.empty() || value.size() > 32U) {
                    throw std::invalid_argument(
                        "session_config " + key +
                        " must contain 1..32 characters"
                    );
                }
                return value;
            };
        auto opponent_character = optional_string("opponent_character");
        auto player_side = optional_string("player_side");
        auto initial_facing = optional_string("initial_facing");
        if (player_side.has_value() &&
            *player_side != "left" && *player_side != "right") {
            throw std::invalid_argument(
                "session_config player_side must be left or right"
            );
        }
        if (initial_facing.has_value() &&
            *initial_facing != "left" && *initial_facing != "right") {
            throw std::invalid_argument(
                "session_config initial_facing must be left or right"
            );
        }
        return SessionConfiguration{
            .player_character = player_character,
            .opponent_character = std::move(opponent_character),
            .player_side = std::move(player_side),
            .initial_facing = std::move(initial_facing),
        };
    }

    if (type == "action_ack") {
        if (!request.contains("protocol_version") ||
            !request["protocol_version"].is_number_integer() ||
            request["protocol_version"].get<int>() !=
                static_cast<int>(kActionProtocolVersion)) {
            throw std::invalid_argument("action_ack protocol_version must be 2");
        }
        if (!request.contains("decision_id") ||
            !request["decision_id"].is_number_unsigned() ||
            request["decision_id"].get<std::uint64_t>() == 0) {
            throw std::invalid_argument("action_ack decision_id must be a positive uint64");
        }
        if (!request.contains("applied_timestamp_ms") ||
            !request["applied_timestamp_ms"].is_number_integer() ||
            request["applied_timestamp_ms"].get<std::int64_t>() <= 0) {
            throw std::invalid_argument("action_ack applied_timestamp_ms must be positive");
        }
        if (!request.contains("applied_commands") ||
            !request["applied_commands"].is_array()) {
            throw std::invalid_argument("action_ack applied_commands must be an array");
        }

        std::array<bool, 10> seen{};
        std::vector<int> commands;
        commands.reserve(request["applied_commands"].size());
        for (const auto& value : request["applied_commands"]) {
            if (!value.is_number_integer()) {
                throw std::invalid_argument("action_ack commands must be integers");
            }
            const int command = value.get<int>();
            if (command < 0 || command >= static_cast<int>(seen.size())) {
                throw std::invalid_argument("action_ack command is outside the 0..9 range");
            }
            if (seen[static_cast<std::size_t>(command)]) {
                throw std::invalid_argument("action_ack commands must be unique");
            }
            seen[static_cast<std::size_t>(command)] = true;
            commands.push_back(command);
        }
        return ActionAcknowledgement{
            .decision_id = request["decision_id"].get<std::uint64_t>(),
            .applied_timestamp_ms = request["applied_timestamp_ms"].get<std::int64_t>(),
            .applied_commands = std::move(commands),
        };
    }

    if (type == "training_reset_ack") {
        if (!request.contains("protocol_version") ||
            !request["protocol_version"].is_number_integer() ||
            request["protocol_version"].get<int>() !=
                static_cast<int>(kActionProtocolVersion)) {
            throw std::invalid_argument(
                "training_reset_ack protocol_version must be 2"
            );
        }
        if (!request.contains("reset_id") ||
            !request["reset_id"].is_number_unsigned() ||
            request["reset_id"].get<std::uint64_t>() == 0) {
            throw std::invalid_argument(
                "training_reset_ack reset_id must be a positive uint64"
            );
        }
        if (!request.contains("completed_timestamp_ms") ||
            !request["completed_timestamp_ms"].is_number_integer() ||
            request["completed_timestamp_ms"].get<std::int64_t>() <= 0) {
            throw std::invalid_argument(
                "training_reset_ack completed_timestamp_ms must be positive"
            );
        }
        if (!request.contains("success") ||
            !request["success"].is_boolean()) {
            throw std::invalid_argument(
                "training_reset_ack success must be a boolean"
            );
        }
        std::optional<std::string> message;
        if (request.contains("message")) {
            if (!request["message"].is_string()) {
                throw std::invalid_argument(
                    "training_reset_ack message must be a string"
                );
            }
            message = request["message"].get<std::string>();
            if (message->size() > 512U) {
                throw std::invalid_argument(
                    "training_reset_ack message is too large"
                );
            }
        }
        return TrainingResetAcknowledgement{
            .reset_id = request["reset_id"].get<std::uint64_t>(),
            .completed_timestamp_ms =
                request["completed_timestamp_ms"].get<std::int64_t>(),
            .success = request["success"].get<bool>(),
            .message = std::move(message),
        };
    }

    if (type != "frame" || !request.contains("frame_id") ||
        !request["frame_id"].is_number_unsigned() ||
        !request.contains("timestamp_ms") || !request["timestamp_ms"].is_number_integer() ||
        !request.contains("jpeg_base64") || !request["jpeg_base64"].is_string()) {
        throw std::invalid_argument(
            "frame message must contain unsigned frame_id, integer timestamp_ms and jpeg_base64"
        );
    }
    return FrameEnvelope{
        .frame_id = request["frame_id"].get<std::uint64_t>(),
        .timestamp_ms = request["timestamp_ms"].get<std::int64_t>(),
        .jpeg = decode_base64(request["jpeg_base64"].get_ref<const std::string&>()),
    };
}

std::string websocket_accept_key(std::string_view client_key) {
    constexpr std::string_view guid = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string input(client_key);
    input.append(guid);
    const auto digest = sha1(input);
    return encode_base64(digest);
}

std::vector<std::uint8_t> make_websocket_server_frame(
    std::uint8_t opcode,
    std::span<const std::uint8_t> payload
) {
    std::vector<std::uint8_t> frame;
    frame.reserve(payload.size() + 10U);
    frame.push_back(static_cast<std::uint8_t>(0x80U | (opcode & 0x0fU)));
    if (payload.size() <= 125U) {
        frame.push_back(static_cast<std::uint8_t>(payload.size()));
    } else if (payload.size() <= 0xffffU) {
        frame.push_back(126U);
        frame.push_back(static_cast<std::uint8_t>((payload.size() >> 8U) & 0xffU));
        frame.push_back(static_cast<std::uint8_t>(payload.size() & 0xffU));
    } else {
        frame.push_back(127U);
        append_u64_be(frame, static_cast<std::uint64_t>(payload.size()));
    }
    frame.insert(frame.end(), payload.begin(), payload.end());
    return frame;
}

}  // namespace gameagent
