#include "MockProtocol.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#include <unistd.h>

namespace gameagent {
namespace {

struct DecodedFrame {
    std::int64_t timestamp_ms = 0;
    std::vector<std::uint8_t> jpeg;
};

ProtocolResponse error_response(int status, std::string message) {
    return ProtocolResponse{
        .http_status = status,
        .body = nlohmann::json{{"error", std::move(message)}},
        .capture_directory = {},
    };
}

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
    constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    if (encoded.starts_with("data:")) {
        const std::size_t comma = encoded.find(',');
        if (comma == std::string_view::npos) {
            throw std::invalid_argument("invalid Base64 data URL");
        }
        encoded.remove_prefix(comma + 1);
    }

    std::string compact;
    compact.reserve(encoded.size());
    for (const unsigned char character : encoded) {
        if (!std::isspace(character)) {
            compact.push_back(static_cast<char>(character));
        }
    }

    if (compact.empty() || compact.size() % 4 != 0) {
        throw std::invalid_argument("invalid Base64 length");
    }

    const auto value_of = [&](char character) -> int {
        const std::size_t position = alphabet.find(character);
        if (position == std::string_view::npos) {
            throw std::invalid_argument("invalid Base64 character");
        }
        return static_cast<int>(position);
    };

    std::vector<std::uint8_t> decoded;
    decoded.reserve((compact.size() / 4) * 3);

    for (std::size_t offset = 0; offset < compact.size(); offset += 4) {
        const bool final_block = offset + 4 == compact.size();
        const char c0 = compact[offset];
        const char c1 = compact[offset + 1];
        const char c2 = compact[offset + 2];
        const char c3 = compact[offset + 3];

        if (c0 == '=' || c1 == '=') {
            throw std::invalid_argument("invalid Base64 padding");
        }

        const bool padding2 = c2 == '=';
        const bool padding3 = c3 == '=';
        if ((!final_block && (padding2 || padding3)) || (padding2 && !padding3)) {
            throw std::invalid_argument("invalid Base64 padding");
        }

        const std::uint32_t v0 = static_cast<std::uint32_t>(value_of(c0));
        const std::uint32_t v1 = static_cast<std::uint32_t>(value_of(c1));
        const std::uint32_t v2 = padding2 ? 0U : static_cast<std::uint32_t>(value_of(c2));
        const std::uint32_t v3 = padding3 ? 0U : static_cast<std::uint32_t>(value_of(c3));
        const std::uint32_t block = (v0 << 18U) | (v1 << 12U) | (v2 << 6U) | v3;

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

bool looks_like_jpeg(const std::vector<std::uint8_t>& bytes) {
    return bytes.size() >= 4 && bytes[0] == 0xffU && bytes[1] == 0xd8U;
}

void write_binary_file(
    const std::filesystem::path& path,
    const std::vector<std::uint8_t>& bytes
) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not open " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
    );
    if (!output) {
        throw std::runtime_error("could not write " + path.string());
    }
}

void write_json_file(const std::filesystem::path& path, const nlohmann::json& value) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("could not open " + path.string());
    }
    output << value.dump(2) << '\n';
    if (!output) {
        throw std::runtime_error("could not write " + path.string());
    }
}

std::string frame_filename(std::size_t index) {
    std::ostringstream name;
    name << "frame_" << std::setw(3) << std::setfill('0') << index << ".jpg";
    return name.str();
}

}  // namespace

MockProtocol::MockProtocol(
    std::filesystem::path capture_root,
    std::optional<std::uint32_t> random_seed
)
    : capture_root_(std::move(capture_root)),
      random_engine_(random_seed.value_or(std::random_device{}())) {
    if (capture_root_.empty()) {
        throw std::invalid_argument("capture directory cannot be empty");
    }
    std::filesystem::create_directories(capture_root_);
}

ProtocolResponse MockProtocol::process(std::string_view request_body) {
    nlohmann::json request;
    try {
        request = nlohmann::json::parse(request_body.begin(), request_body.end());
    } catch (const std::exception& error) {
        return error_response(400, std::string("invalid JSON: ") + error.what());
    }

    if (!request.is_object()) {
        return error_response(400, "request must be a JSON object");
    }
    if (!request.contains("protocol_version") ||
        !request["protocol_version"].is_number_integer() ||
        request["protocol_version"].get<int>() != 1) {
        return error_response(422, "protocol_version must be 1");
    }
    if (!request.contains("emulator") || !request["emulator"].is_string() ||
        request["emulator"].get_ref<const std::string&>().empty()) {
        return error_response(422, "emulator must be a non-empty string");
    }
    if (!request.contains("window_seconds") || !request["window_seconds"].is_number()) {
        return error_response(422, "window_seconds must be numeric");
    }
    if (!request.contains("sample_fps") || !request["sample_fps"].is_number()) {
        return error_response(422, "sample_fps must be numeric");
    }
    if (!request.contains("frames") || !request["frames"].is_array()) {
        return error_response(422, "frames must be an array");
    }
    if (request["frames"].size() != kExpectedFrameCount) {
        return ProtocolResponse{
            .http_status = 422,
            .body = nlohmann::json{
                {"error", "exactly 30 frames are required"},
                {"expected_frames", kExpectedFrameCount},
                {"received_frames", request["frames"].size()},
            },
            .capture_directory = {},
        };
    }

    std::vector<DecodedFrame> decoded_frames;
    decoded_frames.reserve(kExpectedFrameCount);

    try {
        for (std::size_t index = 0; index < kExpectedFrameCount; ++index) {
            const auto& frame = request["frames"][index];
            if (!frame.is_object() || !frame.contains("timestamp_ms") ||
                !frame["timestamp_ms"].is_number_integer() ||
                !frame.contains("jpeg_base64") || !frame["jpeg_base64"].is_string()) {
                return error_response(
                    422,
                    "frame " + std::to_string(index) +
                        " must contain integer timestamp_ms and string jpeg_base64"
                );
            }

            DecodedFrame decoded;
            decoded.timestamp_ms = frame["timestamp_ms"].get<std::int64_t>();
            decoded.jpeg = decode_base64(frame["jpeg_base64"].get_ref<const std::string&>());
            if (!looks_like_jpeg(decoded.jpeg)) {
                return error_response(
                    422,
                    "frame " + std::to_string(index) + " is not a JPEG image"
                );
            }
            decoded_frames.push_back(std::move(decoded));
        }
    } catch (const std::exception& error) {
        return error_response(422, std::string("invalid frame data: ") + error.what());
    }

    const std::filesystem::path request_directory = capture_root_ / make_request_id();
    try {
        std::filesystem::create_directories(request_directory);

        nlohmann::json metadata{
            {"protocol_version", request["protocol_version"]},
            {"emulator", request["emulator"]},
            {"window_seconds", request["window_seconds"]},
            {"sample_fps", request["sample_fps"]},
            {"received_frames", decoded_frames.size()},
            {"frames", nlohmann::json::array()},
        };

        for (std::size_t index = 0; index < decoded_frames.size(); ++index) {
            const std::string filename = frame_filename(index);
            write_binary_file(request_directory / filename, decoded_frames[index].jpeg);
            metadata["frames"].push_back({
                {"index", index},
                {"timestamp_ms", decoded_frames[index].timestamp_ms},
                {"file", filename},
                {"size_bytes", decoded_frames[index].jpeg.size()},
            });
        }
        write_json_file(request_directory / "metadata.json", metadata);

        nlohmann::json response{{"commands", make_random_commands()}};
        write_json_file(request_directory / "response.json", response);

        return ProtocolResponse{
            .http_status = 200,
            .body = std::move(response),
            .capture_directory = request_directory,
        };
    } catch (const std::exception& error) {
        return error_response(500, std::string("could not save capture: ") + error.what());
    }
}

const std::filesystem::path& MockProtocol::capture_root() const {
    return capture_root_;
}

std::vector<int> MockProtocol::make_random_commands() {
    std::lock_guard lock(random_mutex_);

    std::array<int, 10> available{};
    std::iota(available.begin(), available.end(), 0);
    std::shuffle(available.begin(), available.end(), random_engine_);

    std::uniform_int_distribution<std::size_t> count_distribution(3, 10);
    const std::size_t count = count_distribution(random_engine_);
    return std::vector<int>(available.begin(), available.begin() + count);
}

std::string MockProtocol::make_request_id() {
    const auto now = std::chrono::system_clock::now();
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::tm utc{};
    gmtime_r(&time, &utc);

    std::ostringstream identifier;
    identifier << "request_" << std::put_time(&utc, "%Y%m%dT%H%M%S") << '_'
               << (milliseconds % 1000) << "_p" << getpid() << '_'
               << request_counter_.fetch_add(1);
    return identifier.str();
}

}  // namespace gameagent
