#include "WebSocketServer.h"

#include "ActionHistory.h"
#include "CombatStateTracker.h"
#include "FrameBuffer.h"
#include "HealthBarDetector.h"
#include "MockAgent.h"
#include "ReinforcementAgent.h"
#include "RepeatedFrameDetector.h"
#include "StreamProtocol.h"
#include "TemporalAgent.h"

#include <json.hpp>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace gameagent {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kMaximumHeaderBytes = 64U * 1024U;
constexpr std::size_t kMaximumFragmentOverhead = 1024U * 1024U;
// A reset blocks normal actions.  Waiting for a second failed reset used to
// leave the client without controls for roughly 15-20 seconds.  One timeout
// is enough evidence that the reset handshake cannot make progress on the
// current socket; reconnecting is the safe recovery because it also releases
// every controller input.
constexpr std::size_t kResetFailuresBeforeReconnect = 1U;

std::string displayed_action_name(
    std::string_view base_name,
    bool mirrored
) {
    std::string result(base_name);
    if (mirrored && !result.empty() && result != "release") {
        result += " (E)";
    }
    return result;
}

std::string_view controller_state_kind(const std::vector<int>& commands) {
    if (commands.empty()) {
        return "neutral";
    }
    const bool has_direction = std::ranges::any_of(commands, [](int command) {
        return command >= 6 && command <= 9;
    });
    const bool has_button = std::ranges::any_of(commands, [](int command) {
        return command >= 0 && command <= 5;
    });
    if (has_direction && has_button) {
        return "direction_button_chord";
    }
    return has_direction ? "direction" : "buttons";
}

// The reinforcement checkpoint fingerprints its compact option programs.  The
// controller program sent to a client is intentionally a separate, explicit
// representation: a special's terminal direction-button chord is declared by
// the move profile, rather than inferred by the transport from adjacent steps.
struct TimedChordStep {
    std::string_view option_prefix;
    std::size_t command_step;
    std::size_t direction_step;
};

bool has_option_prefix(std::string_view option_name, std::string_view prefix) {
    return option_name == prefix ||
        (option_name.size() > prefix.size() &&
         option_name.starts_with(prefix) && option_name[prefix.size()] == ' ');
}

std::vector<std::vector<int>> make_explicit_controller_program(
    std::string_view option_name,
    const std::vector<std::vector<int>>& compact_program
) {
    static constexpr std::array<TimedChordStep, 13> kChordSteps{{
        {"Congelar", 2U, 1U},
        {"Clone de Gelo", 2U, 1U},
        {"Chuva de Gelo (Frente)", 2U, 1U},
        {"Chuva de Gelo (Perto)", 3U, 2U},
        {"Arremesso de Sombra", 3U, 2U},
        {"Nuvem de Desativação", 2U, 1U},
        {"Arpão", 3U, 2U},
        {"Soco Teleportado (Aéreo)", 5U, 4U},
        {"Soco Teleportado", 2U, 1U},
        {"Bola de Fogo (Aérea)", 6U, 5U},
        {"Bola de Fogo", 3U, 2U},
        {"Grito", 5U, 4U},
        {"Levitar", 4U, 3U},
    }};

    std::vector<std::vector<int>> result = compact_program;
    for (const TimedChordStep& chord : kChordSteps) {
        if (!has_option_prefix(option_name, chord.option_prefix)) {
            continue;
        }
        if (chord.command_step >= result.size() ||
            chord.direction_step >= result.size()) {
            throw std::logic_error(
                "timed controller program does not match its move profile"
            );
        }
        for (int command : result[chord.direction_step]) {
            if (command < 6 || command > 9 ||
                std::ranges::find(result[chord.command_step], command) !=
                    result[chord.command_step].end()) {
                continue;
            }
            result[chord.command_step].insert(
                result[chord.command_step].begin(), command
            );
        }
        break;
    }
    return result;
}

bool requires_external_encounter_lifecycle(
    const EncounterConfiguration& encounter
) {
    return encounter.type == EncounterType::endurance &&
        encounter.current_stage + 1U < encounter.opponent_count;
}

ResetReason terminal_reset_reason_from_trigger(
    std::string_view trigger_reason,
    bool compatibility_recovery
) {
    if (trigger_reason == "round_won" ||
        trigger_reason == "encounter_won") {
        return ResetReason::encounter_completed;
    }
    if (trigger_reason == "round_lost" ||
        trigger_reason == "encounter_lost") {
        return ResetReason::player_defeated;
    }
    return compatibility_recovery
        ? ResetReason::stream_stalled
        : ResetReason::invalid_game_state;
}

struct SocketHandle {
    int value = -1;

    SocketHandle() = default;
    explicit SocketHandle(int descriptor) : value(descriptor) {}
    SocketHandle(const SocketHandle&) = delete;
    SocketHandle& operator=(const SocketHandle&) = delete;
    SocketHandle(SocketHandle&& other) noexcept : value(std::exchange(other.value, -1)) {}
    SocketHandle& operator=(SocketHandle&& other) noexcept {
        if (this != &other) {
            if (value >= 0) {
                close(value);
            }
            value = std::exchange(other.value, -1);
        }
        return *this;
    }
    ~SocketHandle() {
        if (value >= 0) {
            close(value);
        }
    }
};

struct HttpRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
};

struct WebSocketFrame {
    bool final = false;
    std::uint8_t opcode = 0;
    std::vector<std::uint8_t> payload;
};

std::int64_t epoch_milliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

nlohmann::json make_timed_sequence_payload(
    std::uint64_t sequence_id,
    std::string_view option_name,
    const std::vector<std::vector<int>>& commands,
    const WebSocketServerOptions& options
) {
    const std::vector<std::vector<int>> controller_program =
        make_explicit_controller_program(option_name, commands);
    nlohmann::json steps = nlohmann::json::array();
    std::int64_t total_duration_ms = 0;
    for (std::size_t index = 0; index < controller_program.size(); ++index) {
        const bool final_step = index + 1U == controller_program.size();
        const std::int64_t duration_ms = final_step
            ? options.sequence_final_release.count()
            : controller_program[index].empty()
                ? options.sequence_neutral_hold.count()
                : options.sequence_input_hold.count();
        steps.push_back({
            {"step_index", index + 1U},
            {"commands", controller_program[index]},
            {"state_kind", controller_state_kind(controller_program[index])},
            {"apply", "replace_state"},
            {"duration_ms", duration_ms},
        });
        total_duration_ms += duration_ms;
    }
    if (controller_program.empty() || !controller_program.back().empty()) {
        const std::int64_t duration_ms =
            options.sequence_final_release.count();
        steps.push_back({
            {"step_index", steps.size() + 1U},
            {"commands", nlohmann::json::array()},
            {"state_kind", "neutral"},
            {"apply", "replace_state"},
            {"duration_ms", duration_ms},
        });
        total_duration_ms += duration_ms;
    }
    return {
        {"version", 2},
        {"sequence_id", sequence_id},
        {"option_name", option_name},
        {"start_policy", "replace"},
        {"step_semantics", "exact_controller_state"},
        {"transition_policy", "replace_state"},
        {"clock", "monotonic"},
        {"release_on_complete", true},
        {"step_count", steps.size()},
        {"total_duration_ms", total_duration_ms},
        {"steps", std::move(steps)},
    };
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string trim(std::string value) {
    const auto not_space = [](unsigned char character) {
        return !std::isspace(character);
    };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

bool constant_time_equals(std::string_view left, std::string_view right) {
    const std::size_t size = std::max(left.size(), right.size());
    std::size_t difference = left.size() ^ right.size();
    for (std::size_t index = 0; index < size; ++index) {
        const unsigned char a = index < left.size() ? left[index] : 0U;
        const unsigned char b = index < right.size() ? right[index] : 0U;
        difference |= static_cast<std::size_t>(a ^ b);
    }
    return difference == 0;
}

bool request_is_authorized(
    const HttpRequest& request,
    const std::optional<std::string>& api_token
) {
    if (!api_token.has_value()) {
        return true;
    }
    if (const auto key = request.headers.find("x-api-key");
        key != request.headers.end() && constant_time_equals(key->second, *api_token)) {
        return true;
    }
    const auto authorization = request.headers.find("authorization");
    constexpr std::string_view prefix = "Bearer ";
    return authorization != request.headers.end() &&
        authorization->second.starts_with(prefix) &&
        constant_time_equals(
            std::string_view(authorization->second).substr(prefix.size()),
            *api_token
        );
}

bool send_all(int socket, std::span<const std::uint8_t> bytes) {
    std::size_t sent = 0;
    while (sent < bytes.size()) {
        const ssize_t result = send(
            socket,
            bytes.data() + sent,
            bytes.size() - sent,
#ifdef MSG_NOSIGNAL
            MSG_NOSIGNAL
#else
            0
#endif
        );
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (result == 0) {
            return false;
        }
        sent += static_cast<std::size_t>(result);
    }
    return true;
}

bool send_text(int socket, std::string_view text) {
    return send_all(
        socket,
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(text.data()),
            text.size()
        )
    );
}

void send_http_json(int socket, int status, std::string_view reason, const nlohmann::json& body) {
    const std::string payload = body.dump();
    std::ostringstream response;
    response << "HTTP/1.1 " << status << ' ' << reason << "\r\n"
             << "Content-Type: application/json; charset=utf-8\r\n"
             << "Content-Length: " << payload.size() << "\r\n"
             << "Access-Control-Allow-Origin: *\r\n"
             << "Connection: close\r\n\r\n"
             << payload;
    send_text(socket, response.str());
}

class SocketReader {
public:
    explicit SocketReader(int socket) : socket_(socket) {}

    bool read_http_headers(std::string& headers, const std::atomic_bool& stop_requested) {
        constexpr std::string_view delimiter = "\r\n\r\n";
        while (true) {
            const auto position = std::search(
                buffered_.begin(), buffered_.end(), delimiter.begin(), delimiter.end()
            );
            if (position != buffered_.end()) {
                const auto end = position + static_cast<std::ptrdiff_t>(delimiter.size());
                headers.assign(buffered_.begin(), end);
                buffered_.erase(buffered_.begin(), end);
                return true;
            }
            if (buffered_.size() >= kMaximumHeaderBytes || !receive_more(stop_requested)) {
                return false;
            }
        }
    }

    bool read_exact(
        std::span<std::uint8_t> destination,
        const std::atomic_bool& stop_requested
    ) {
        std::size_t copied = 0;
        if (!buffered_.empty()) {
            const std::size_t count = std::min(buffered_.size(), destination.size());
            std::copy_n(buffered_.begin(), count, destination.begin());
            buffered_.erase(buffered_.begin(), buffered_.begin() + static_cast<std::ptrdiff_t>(count));
            copied = count;
        }
        while (copied < destination.size()) {
            if (!wait_readable(stop_requested)) {
                return false;
            }
            const ssize_t received = recv(
                socket_,
                destination.data() + copied,
                destination.size() - copied,
                0
            );
            if (received < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (received == 0) {
                return false;
            }
            copied += static_cast<std::size_t>(received);
        }
        return true;
    }

private:
    bool wait_readable(const std::atomic_bool& stop_requested) const {
        while (!stop_requested.load()) {
            pollfd descriptor{.fd = socket_, .events = POLLIN, .revents = 0};
            const int result = poll(&descriptor, 1, 250);
            if (result < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            if (result == 0) {
                continue;
            }
            return (descriptor.revents & POLLIN) != 0;
        }
        return false;
    }

    bool receive_more(const std::atomic_bool& stop_requested) {
        if (!wait_readable(stop_requested)) {
            return false;
        }
        std::array<std::uint8_t, 8192> incoming{};
        const ssize_t received = recv(socket_, incoming.data(), incoming.size(), 0);
        if (received <= 0) {
            return false;
        }
        buffered_.insert(buffered_.end(), incoming.begin(), incoming.begin() + received);
        return true;
    }

    int socket_;
    std::vector<std::uint8_t> buffered_;
};

HttpRequest parse_http_request(std::string_view raw) {
    std::istringstream stream{std::string(raw)};
    std::string request_line;
    if (!std::getline(stream, request_line)) {
        throw std::invalid_argument("missing HTTP request line");
    }
    if (!request_line.empty() && request_line.back() == '\r') {
        request_line.pop_back();
    }
    HttpRequest request;
    std::string version;
    std::istringstream request_line_stream(request_line);
    if (!(request_line_stream >> request.method >> request.path >> version) ||
        version != "HTTP/1.1") {
        throw std::invalid_argument("invalid HTTP/1.1 request line");
    }

    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            break;
        }
        const std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            throw std::invalid_argument("invalid HTTP header");
        }
        request.headers[lowercase(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1U));
    }
    return request;
}

std::string route_path(std::string_view path) {
    const std::size_t query = path.find('?');
    return std::string(path.substr(0, query));
}

bool header_contains_token(const HttpRequest& request, std::string_view name, std::string token) {
    const auto iterator = request.headers.find(std::string(name));
    if (iterator == request.headers.end()) {
        return false;
    }
    const std::string value = lowercase(iterator->second);
    token = lowercase(std::move(token));
    std::size_t begin = 0;
    while (begin < value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::size_t end = comma == std::string::npos ? value.size() : comma;
        if (trim(value.substr(begin, end - begin)) == token) {
            return true;
        }
        begin = end + 1U;
    }
    return false;
}

bool perform_websocket_upgrade(
    int socket,
    const HttpRequest& request,
    const std::optional<std::string>& api_token
) {
    if (request.method != "GET" || route_path(request.path) != "/ws") {
        send_http_json(socket, 404, "Not Found", {{"error", "route not found"}});
        return false;
    }
    if (!request_is_authorized(request, api_token)) {
        send_http_json(socket, 401, "Unauthorized", {{"error", "missing or invalid API token"}});
        return false;
    }
    const auto version = request.headers.find("sec-websocket-version");
    const auto key = request.headers.find("sec-websocket-key");
    if (!header_contains_token(request, "upgrade", "websocket") ||
        !header_contains_token(request, "connection", "upgrade") ||
        version == request.headers.end() || version->second != "13" ||
        key == request.headers.end() || key->second.empty()) {
        send_http_json(socket, 400, "Bad Request", {{"error", "invalid WebSocket upgrade"}});
        return false;
    }

    std::ostringstream response;
    response << "HTTP/1.1 101 Switching Protocols\r\n"
             << "Upgrade: websocket\r\n"
             << "Connection: Upgrade\r\n"
             << "Sec-WebSocket-Accept: " << websocket_accept_key(key->second) << "\r\n\r\n";
    return send_text(socket, response.str());
}

std::uint64_t read_u64_be(std::span<const std::uint8_t> bytes) {
    std::uint64_t value = 0;
    for (std::uint8_t byte : bytes) {
        value = (value << 8U) | byte;
    }
    return value;
}

bool read_websocket_frame(
    SocketReader& reader,
    WebSocketFrame& frame,
    std::size_t maximum_payload,
    const std::atomic_bool& stop_requested
) {
    std::array<std::uint8_t, 2> header{};
    if (!reader.read_exact(header, stop_requested)) {
        return false;
    }
    frame.final = (header[0] & 0x80U) != 0;
    const bool reserved = (header[0] & 0x70U) != 0;
    frame.opcode = header[0] & 0x0fU;
    const bool masked = (header[1] & 0x80U) != 0;
    std::uint64_t payload_size = header[1] & 0x7fU;
    if (reserved || !masked) {
        throw std::invalid_argument("client WebSocket frames must be masked and have no RSV bits");
    }
    if (payload_size == 126U) {
        std::array<std::uint8_t, 2> extended{};
        if (!reader.read_exact(extended, stop_requested)) {
            return false;
        }
        payload_size = (static_cast<std::uint64_t>(extended[0]) << 8U) | extended[1];
    } else if (payload_size == 127U) {
        std::array<std::uint8_t, 8> extended{};
        if (!reader.read_exact(extended, stop_requested)) {
            return false;
        }
        if ((extended[0] & 0x80U) != 0) {
            throw std::invalid_argument("invalid 64-bit WebSocket payload length");
        }
        payload_size = read_u64_be(extended);
    }
    const bool control = (frame.opcode & 0x08U) != 0;
    if ((control && (!frame.final || payload_size > 125U)) || payload_size > maximum_payload) {
        throw std::invalid_argument("WebSocket payload exceeds configured limit");
    }

    std::array<std::uint8_t, 4> mask{};
    if (!reader.read_exact(mask, stop_requested)) {
        return false;
    }
    frame.payload.resize(static_cast<std::size_t>(payload_size));
    if (!frame.payload.empty() && !reader.read_exact(frame.payload, stop_requested)) {
        return false;
    }
    for (std::size_t index = 0; index < frame.payload.size(); ++index) {
        frame.payload[index] ^= mask[index % mask.size()];
    }
    return true;
}

class ConnectionWriter {
public:
    explicit ConnectionWriter(int socket) : socket_(socket) {}

    bool send_frame(std::uint8_t opcode, std::span<const std::uint8_t> payload) {
        const std::vector<std::uint8_t> frame = make_websocket_server_frame(opcode, payload);
        std::lock_guard lock(mutex_);
        return send_all(socket_, frame);
    }

    bool send_json(const nlohmann::json& value) {
        const std::string text = value.dump();
        return send_frame(
            0x1U,
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(text.data()), text.size()
            )
        );
    }

private:
    int socket_;
    std::mutex mutex_;
};

void send_close(ConnectionWriter& writer, std::uint16_t code, std::string_view message) {
    const std::size_t message_size = std::min<std::size_t>(message.size(), 123U);
    std::vector<std::uint8_t> payload;
    payload.reserve(message_size + 2U);
    payload.push_back(static_cast<std::uint8_t>((code >> 8U) & 0xffU));
    payload.push_back(static_cast<std::uint8_t>(code & 0xffU));
    payload.insert(payload.end(), message.begin(), message.begin() + static_cast<std::ptrdiff_t>(message_size));
    writer.send_frame(0x8U, payload);
}

std::shared_ptr<const BufferedFrame> decode_frame(
    const FrameEnvelope& envelope,
    std::size_t maximum_jpeg_bytes,
    std::int64_t received_timestamp_ms
) {
    if (envelope.jpeg.empty() || envelope.jpeg.size() > maximum_jpeg_bytes ||
        envelope.jpeg.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("JPEG exceeds configured size limit");
    }
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* decoded = stbi_load_from_memory(
        envelope.jpeg.data(),
        static_cast<int>(envelope.jpeg.size()),
        &width,
        &height,
        &channels,
        3
    );
    if (decoded == nullptr) {
        throw std::invalid_argument(
            std::string("JPEG decode failed: ") +
            (stbi_failure_reason() == nullptr ? "unknown error" : stbi_failure_reason())
        );
    }
    struct ImageGuard {
        unsigned char* value;
        ~ImageGuard() { stbi_image_free(value); }
    } guard{decoded};
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("JPEG has invalid dimensions");
    }

    base::data::image::Image image(width, height, 3);
    std::memcpy(image.data(), decoded, image.size_bytes());
    return std::make_shared<BufferedFrame>(BufferedFrame{
        .frame_id = envelope.frame_id,
        .client_timestamp_ms = envelope.timestamp_ms,
        .received_timestamp_ms = received_timestamp_ms,
        .jpeg_size = envelope.jpeg.size(),
        .jpeg = envelope.jpeg,
        .image = std::move(image),
    });
}

class MetricsLogger {
public:
    explicit MetricsLogger(const std::filesystem::path& path) {
        if (path.empty()) {
            return;
        }
        if (!path.parent_path().empty()) {
            std::filesystem::create_directories(path.parent_path());
        }
        output_.open(path, std::ios::app);
        if (!output_) {
            throw std::runtime_error("could not open metrics file: " + path.string());
        }
    }

    void write(const nlohmann::json& record) {
        std::lock_guard lock(mutex_);
        if (output_) {
            output_ << record.dump() << '\n';
            ++pending_records_;
            const bool lifecycle_event = record.value("event", "") != "decision";
            if (lifecycle_event || pending_records_ >= 20U) {
                output_.flush();
                pending_records_ = 0;
            }
        }
    }

private:
    std::ofstream output_;
    std::mutex mutex_;
    std::size_t pending_records_ = 0;
};

DirectedActionCurriculum load_directed_curriculum(
    const std::filesystem::path& path,
    FighterCharacter character
) {
    DirectedActionCurriculum result;
    if (path.empty() || !std::filesystem::is_regular_file(path)) {
        return result;
    }
    std::ifstream input(path);
    nlohmann::json document;
    input >> document;
    result.enabled = document.value("enabled", false);
    result.include_mirrors = document.value("include_mirrors", true);
    const std::string mode = document.value("mode", "exploration_only");
    if (mode != "exploration_only" && mode != "force_only") {
        throw std::invalid_argument(
            "directed curriculum mode must be exploration_only or force_only"
        );
    }
    result.force_only = mode == "force_only";
    if (!result.enabled) {
        return result;
    }
    const nlohmann::json& characters = document.at("characters");
    const std::string character_key(fighter_character_slug(character));
    const auto selected = characters.contains(character_key)
        ? characters.at(character_key)
        : characters.contains("*") ? characters.at("*")
        : nlohmann::json::array();
    if (!selected.is_array()) {
        throw std::invalid_argument(
            "directed curriculum character moves must be an array"
        );
    }
    for (const auto& value : selected) {
        if (!value.is_string()) {
            throw std::invalid_argument(
                "directed curriculum move names must be strings"
            );
        }
        result.move_names.push_back(value.get<std::string>());
    }
    if (result.move_names.empty()) {
        throw std::invalid_argument(
            "directed curriculum has no moves for " + character_key
        );
    }
    return result;
}

struct FramePersistenceJob {
    std::shared_ptr<const FrameEnvelope> envelope;
    std::int64_t received_timestamp_ms = 0;
    std::optional<AppliedActionRecord> latest_applied_action;
    std::string player_character;
    std::string opponent_character;
    bool training_capture = false;
};

void save_frame(
    const WebSocketServerOptions& options,
    std::uint64_t connection_id,
    const FramePersistenceJob& job
) {
    const FrameEnvelope& envelope = *job.envelope;
    const std::filesystem::path& capture_root =
        job.training_capture
            ? options.training_capture_directory
            : options.capture_directory;
    const std::filesystem::path directory = capture_root /
        ("session_" + std::to_string(getpid())) /
        ("connection_" + std::to_string(connection_id));
    std::filesystem::create_directories(directory);
    const std::string filename =
        ("frame_" + std::to_string(envelope.frame_id) + ".jpg");
    const std::filesystem::path path = directory / filename;
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("could not save frame: " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(envelope.jpeg.data()),
        static_cast<std::streamsize>(envelope.jpeg.size())
    );
    if (!output) {
        throw std::runtime_error("could not save frame: " + path.string());
    }
    std::ofstream metadata(directory / "frames.jsonl", std::ios::app);
    if (!metadata) {
        throw std::runtime_error("could not save frame metadata in: " + directory.string());
    }
    nlohmann::json metadata_record{
        {"frame_id", envelope.frame_id},
        {"client_timestamp_ms", envelope.timestamp_ms},
        {"received_timestamp_ms", job.received_timestamp_ms},
        {"jpeg_bytes", envelope.jpeg.size()},
        {"file", filename},
        {"capture_mode",
         job.training_capture ? "training" : "debug"},
        {"player_character", job.player_character},
        {"opponent_character", job.opponent_character},
        {"applied_command_known", job.latest_applied_action.has_value()},
    };
    if (job.latest_applied_action.has_value()) {
        metadata_record["applied_decision_id"] =
            job.latest_applied_action->decision_id;
        metadata_record["applied_commands"] =
            job.latest_applied_action->commands;
        metadata_record["applied_timestamp_ms"] =
            job.latest_applied_action->applied_timestamp_ms;
        metadata_record["acknowledgement_received_timestamp_ms"] =
            job.latest_applied_action
                ->acknowledgement_received_timestamp_ms;
    }
    metadata << metadata_record.dump() << '\n';
    if (!metadata) {
        throw std::runtime_error("could not save frame metadata in: " + directory.string());
    }
}

std::optional<AppliedActionRecord> latest_applied_record(
    const ActionHistory& action_history
) {
    const auto actions = action_history.latest_applied(1);
    return actions.empty()
        ? std::nullopt
        : std::optional<AppliedActionRecord>(actions.back());
}

std::vector<float> make_reinforcement_state(
    const TemporalAgentOutput& temporal,
    const FightControlSnapshot& fight,
    const CombatStateSnapshot& combat,
    const std::optional<AppliedActionRecord>& latest_applied,
    const EncounterConfiguration& encounter
) {
    std::vector<float> state;
    state.reserve(
        temporal.lstm.hidden.size() +
        kReinforcementContextFeatureCount
    );
    state.insert(
        state.end(), temporal.lstm.hidden.begin(), temporal.lstm.hidden.end()
    );
    state.push_back(std::clamp(fight.player_health, 0.0f, 1.0f));
    state.push_back(std::clamp(fight.opponent_health, 0.0f, 1.0f));
    state.push_back(std::clamp(
        fight.player_health - fight.opponent_health, -1.0f, 1.0f
    ));
    state.push_back(std::clamp(fight.hud_confidence, 0.0f, 1.0f));
    std::array<float, 10> commands{};
    if (latest_applied.has_value()) {
        commands = latest_applied->command_state;
    }
    state.insert(state.end(), commands.begin(), commands.end());
    state.push_back(latest_applied.has_value() ? 1.0f : 0.0f);
    state.push_back(std::clamp(combat.player_x, 0.0f, 1.0f));
    state.push_back(std::clamp(combat.opponent_x, 0.0f, 1.0f));
    state.push_back(std::clamp(combat.distance, 0.0f, 1.0f));
    state.push_back(combat.player_on_ground ? 1.0f : 0.0f);
    state.push_back(combat.opponent_on_ground ? 1.0f : 0.0f);
    state.push_back(combat.player_defending ? 1.0f : 0.0f);
    state.push_back(combat.opponent_defending ? 1.0f : 0.0f);
    state.push_back(combat.player_attacking ? 1.0f : 0.0f);
    state.push_back(combat.opponent_attacking ? 1.0f : 0.0f);
    state.push_back(std::clamp(
        static_cast<float>(combat.time_since_last_damage_ms) /
            5000.0f,
        0.0f,
        1.0f
    ));
    state.push_back(std::clamp(
        static_cast<float>(combat.consecutive_hits) / 8.0f,
        0.0f,
        1.0f
    ));
    state.push_back(std::clamp(
        static_cast<float>(combat.stationary_time_ms) / 5000.0f,
        0.0f,
        1.0f
    ));
    state.push_back(std::clamp(
        static_cast<float>(combat.corner_time_ms) / 5000.0f,
        0.0f,
        1.0f
    ));
    state.push_back(std::clamp(
        combat.spatial_confidence, 0.0f, 1.0f
    ));
    state.push_back(std::clamp(
        combat.opponent_action_confidence, 0.0f, 1.0f
    ));
    state.push_back(encounter.type == EncounterType::normal ? 1.0f : 0.0f);
    state.push_back(encounter.type == EncounterType::endurance ? 1.0f : 0.0f);
    state.push_back(encounter.type == EncounterType::boss ? 1.0f : 0.0f);
    state.push_back(static_cast<float>(encounter.current_stage) / 3.0f);
    state.push_back(static_cast<float>(encounter.opponent_count) / 4.0f);
    state.push_back(
        static_cast<float>(
            encounter.opponent_count - encounter.current_stage - 1U
        ) / 3.0f
    );
    if (
        state.size() !=
        temporal.lstm.hidden.size() +
            kReinforcementContextFeatureCount
    ) {
        throw std::logic_error(
            "reinforcement combat context size mismatch"
        );
    }
    return state;
}

SocketHandle create_listening_socket(const WebSocketServerOptions& options) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    addrinfo* raw_addresses = nullptr;
    const std::string port = std::to_string(options.port);
    const int lookup = getaddrinfo(
        options.host.empty() ? nullptr : options.host.c_str(),
        port.c_str(),
        &hints,
        &raw_addresses
    );
    if (lookup != 0) {
        throw std::runtime_error("could not resolve bind address: " + std::string(gai_strerror(lookup)));
    }
    struct AddressGuard {
        addrinfo* value;
        ~AddressGuard() { freeaddrinfo(value); }
    } addresses{raw_addresses};

    for (addrinfo* address = raw_addresses; address != nullptr; address = address->ai_next) {
        SocketHandle candidate(socket(address->ai_family, address->ai_socktype, address->ai_protocol));
        if (candidate.value < 0) {
            continue;
        }
        int reuse = 1;
        setsockopt(candidate.value, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        if (bind(candidate.value, address->ai_addr, address->ai_addrlen) == 0 &&
            listen(candidate.value, 16) == 0) {
            return candidate;
        }
    }
    throw std::runtime_error("could not bind WebSocket server to " + options.host + ':' + port);
}

void configure_client_socket(int socket) {
    int enabled = 1;
    setsockopt(socket, IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
    timeval timeout{.tv_sec = 2, .tv_usec = 0};
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

bool wait_for_client(int listening_socket, const std::atomic_bool& stop_requested) {
    while (!stop_requested.load()) {
        pollfd descriptor{.fd = listening_socket, .events = POLLIN, .revents = 0};
        const int result = poll(&descriptor, 1, 250);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("poll failed while waiting for a connection");
        }
        if (result > 0 && (descriptor.revents & POLLIN) != 0) {
            return true;
        }
    }
    return false;
}

void handle_websocket_connection(
    int socket,
    SocketReader& reader,
    const WebSocketServerOptions& options,
    TemporalAgent* temporal_agent,
    std::unique_ptr<ReinforcementAgent>& reinforcement_agent_storage,
    MetricsLogger& metrics,
    std::uint64_t connection_id,
    const std::atomic_bool& stop_requested
) {
    ConnectionWriter writer(socket);
    const std::size_t maximum_payload =
        options.max_jpeg_bytes + kMaximumFragmentOverhead;
    SessionConfiguration session{
        .player_character = "subzero",
        .player_side = options.player_is_left
            ? std::optional<std::string>("left")
            : std::optional<std::string>("right"),
        .initial_facing = options.default_facing_right
            ? std::optional<std::string>("right")
            : std::optional<std::string>("left"),
    };
    EncounterConfiguration encounter_configuration;
    bool v21_session = false;
    MessageMetadata configure_session_metadata;
    bool explicit_session_configuration = false;
    std::optional<WebSocketFrame> pending_first_frame;
    while (!stop_requested.load()) {
        WebSocketFrame candidate;
        if (!read_websocket_frame(
                reader,
                candidate,
                maximum_payload,
                stop_requested
            )) {
            return;
        }
        if (candidate.opcode == 0x8U) {
            writer.send_frame(0x8U, candidate.payload);
            return;
        }
        if (candidate.opcode == 0x9U) {
            if (!writer.send_frame(0xAU, candidate.payload)) {
                return;
            }
            continue;
        }
        if (candidate.opcode == 0xAU) {
            continue;
        }
        if (candidate.opcode == 0x1U && candidate.final) {
            try {
                const std::string text(
                    candidate.payload.begin(), candidate.payload.end()
                );
                ClientTextMessage message =
                    parse_client_text_message(text);
                if (std::holds_alternative<SessionConfiguration>(message)) {
                    session = std::get<SessionConfiguration>(
                        std::move(message)
                    );
                    explicit_session_configuration = true;
                    break;
                }
                if (std::holds_alternative<ConfigureSessionRequest>(message)) {
                    ConfigureSessionRequest request =
                        std::get<ConfigureSessionRequest>(std::move(message));
                    session.player_character = request.player_character;
                    session.opponent_character = std::string(
                        opponent_character_slug(
                            request.encounter.current_opponent()
                        )
                    );
                    session.player_side = std::move(request.player_side);
                    session.initial_facing =
                        std::move(request.initial_facing);
                    encounter_configuration = request.encounter;
                    configure_session_metadata = request.metadata;
                    explicit_session_configuration = true;
                    v21_session = true;
                    break;
                }
            } catch (const std::exception& error) {
                writer.send_json({
                    {"protocol_version", kActionProtocolVersion},
                    {"type", "session_config_error"},
                    {"message", error.what()},
                    {"supported_player_characters",
                     supported_fighter_characters()},
                    {"supported_opponent_characters",
                     supported_opponent_characters()},
                });
                send_close(writer, 1008, error.what());
                return;
            }
        }
        pending_first_frame = std::move(candidate);
        break;
    }

    const auto parsed_character =
        parse_fighter_character(session.player_character);
    if (!parsed_character.has_value()) {
        const std::string message =
            "unsupported player_character: " +
            session.player_character;
        writer.send_json({
            {"protocol_version", kActionProtocolVersion},
            {"type", "session_config_error"},
            {"message", message},
            {"supported_player_characters",
             supported_fighter_characters()},
            {"supported_opponent_characters",
             supported_opponent_characters()},
        });
        send_close(writer, 1008, message);
        return;
    }
    const FighterCharacter player_character = *parsed_character;
    OpponentCharacter opponent_character = OpponentCharacter::unknown;
    if (session.opponent_character.has_value()) {
        const auto parsed_opponent =
            parse_opponent_character(*session.opponent_character);
        if (!parsed_opponent.has_value()) {
            const std::string message =
                "unsupported opponent_character: " +
                *session.opponent_character;
            writer.send_json({
                {"protocol_version", kActionProtocolVersion},
                {"type", "session_config_error"},
                {"message", message},
                {"supported_opponent_characters",
                 supported_opponent_characters()},
            });
            send_close(writer, 1008, message);
            return;
        }
        opponent_character = *parsed_opponent;
    }
    session.opponent_character = std::string(
        opponent_character_slug(opponent_character)
    );
    if (!v21_session) {
        encounter_configuration.type = EncounterType::normal;
        encounter_configuration.opponent_count = 1;
        encounter_configuration.current_stage = 0;
        encounter_configuration.opponents[0] = opponent_character;
    }
    ServerSessionV21 protocol_session(
        options.reinforcement_training_enabled
            ? AgentMode::training
            : AgentMode::evaluation,
        options.reinforcement_epsilon_final
    );
    if (v21_session) {
        protocol_session.configure(encounter_configuration);
    }
    std::mutex protocol_session_mutex;
    std::atomic<OpponentCharacter> active_opponent_character{
        opponent_character
    };
    const bool player_is_left =
        session.player_side.value_or(
            options.player_is_left ? "left" : "right"
        ) == "left";
    const FighterFacing facing =
        session.initial_facing.value_or(
            options.default_facing_right ? "right" : "left"
        ) == "right"
            ? FighterFacing::right
            : FighterFacing::left;
    bool reinforcement_agent_reused = false;
    if (options.reinforcement_learning_enabled) {
        const std::string character_slug(
            fighter_character_slug(player_character)
        );
        const auto latest_checkpoint =
            options.reinforcement_checkpoint_directory /
            (
                options.reinforcement_trainable_lstm
                    ? "shared_recurrent_metadata_latest.garl"
                    : "shared_policy_latest.garl"
            );
        const auto best_checkpoint =
            options.reinforcement_checkpoint_directory /
            (
                options.reinforcement_trainable_lstm
                    ? "shared_recurrent_metadata_best_" +
                          character_slug + ".garl"
                    : "shared_policy_best_" +
                          character_slug + ".garl"
            );
        const auto recurrent_checkpoint =
            options.reinforcement_checkpoint_directory /
            (
                options.reinforcement_backend == "cpp"
                    ? "shared_recurrent_latest.baseai.ckpt"
                    : "shared_recurrent_latest.pt"
            );
        // A no-resume launch is fresh only while the directory has no
        // checkpoint. Once the first connection has persisted progress, every
        // reconnect resumes it automatically instead of silently resetting
        // epsilon, replay and neural weights.
        const bool effective_resume =
            options.reinforcement_resume ||
            std::filesystem::is_regular_file(latest_checkpoint) ||
            std::filesystem::is_regular_file(recurrent_checkpoint);
        try {
            if (reinforcement_agent_storage == nullptr) {
                reinforcement_agent_storage =
                    std::make_unique<ReinforcementAgent>(
                    ReinforcementAgentConfig{
                        .input_size =
                            options.temporal_hidden_size +
                            kReinforcementContextFeatureCount,
                        .network_hidden_size =
                            options.reinforcement_network_hidden_size,
                        .replay_capacity =
                            options.reinforcement_replay_capacity,
                        .minimum_replay_size =
                            options.reinforcement_minimum_replay_size,
                        .batch_size =
                            options.reinforcement_batch_size,
                        .training_updates_per_transition = 1,
                        .target_update_interval =
                            options
                                .reinforcement_target_update_interval,
                        .gamma = options.reinforcement_gamma,
                        .q_value_limit =
                            options.reinforcement_q_value_limit,
                        .terminal_replay_fraction =
                            options
                                .reinforcement_terminal_replay_fraction,
                        .hard_negative_replay_fraction =
                            options
                                .reinforcement_hard_negative_replay_fraction,
                        .winning_replay_fraction =
                            options
                                .reinforcement_winning_replay_fraction,
                        .winning_replay_min_fraction =
                            options
                                .reinforcement_winning_replay_min_fraction,
                        .winning_replay_ramp_episodes =
                            options
                                .reinforcement_winning_replay_ramp_episodes,
                        .winning_replay_capacity =
                            options
                                .reinforcement_winning_replay_capacity,
                        .hard_negative_q_threshold =
                            options
                                .reinforcement_hard_negative_q_threshold,
                        .n_step_options =
                            options.reinforcement_n_step_options,
                        .conservative_bootstrap =
                            options.reinforcement_conservative_bootstrap,
                        .learning_rate =
                            options.reinforcement_learning_rate,
                        .epsilon_start =
                            options.reinforcement_epsilon_start,
                        .epsilon_final =
                            options.reinforcement_epsilon_final,
                        .epsilon_decay_selections =
                            options
                                .reinforcement_epsilon_decay_selections,
                        .new_matchup_epsilon_start =
                            options
                                .reinforcement_new_matchup_epsilon_start,
                        .new_matchup_epsilon_decay_selections =
                            options
                                .reinforcement_new_matchup_epsilon_decay_selections,
                        .scripted_exploration_probability =
                            options
                                .reinforcement_scripted_exploration_probability,
                        .neutral_option_penalty =
                            options.reinforcement_neutral_option_penalty,
                        .suppress_neutral_during_fight =
                            options
                                .reinforcement_suppress_neutral_during_fight,
                        .combo_max_distance =
                            options.reinforcement_combo_max_distance,
                        .close_attack_max_distance =
                            options.reinforcement_close_attack_max_distance,
                        .throw_max_distance =
                            options.reinforcement_throw_max_distance,
                        .sweep_max_distance =
                            options.reinforcement_sweep_max_distance,
                        .projectile_preferred_min_distance =
                            options
                                .reinforcement_projectile_preferred_min_distance,
                        .projectile_q_preference_margin =
                            options
                                .reinforcement_projectile_q_preference_margin,
                        .disable_cloud_min_distance =
                            options.reinforcement_disable_cloud_min_distance,
                        .disable_cloud_max_distance =
                            options.reinforcement_disable_cloud_max_distance,
                        .advance_min_distance =
                            options.reinforcement_advance_min_distance,
                        .defense_max_distance =
                            options.reinforcement_defense_max_distance,
                        .dynamic_facing =
                            options.reinforcement_dynamic_facing,
                        .disable_cloud_cooldown_options =
                            options
                                .reinforcement_disable_cloud_cooldown_options,
                        .throw_cooldown_options =
                            options.reinforcement_throw_cooldown_options,
                        .sweep_cooldown_options =
                            options.reinforcement_sweep_cooldown_options,
                        .melee_impact_credit_steps =
                            options.reinforcement_melee_impact_credit_steps,
                        .teleport_impact_credit_steps =
                            options.reinforcement_teleport_impact_credit_steps,
                        .projectile_impact_credit_min_steps =
                            options
                                .reinforcement_projectile_impact_credit_min_steps,
                        .projectile_impact_credit_max_steps =
                            options
                                .reinforcement_projectile_impact_credit_max_steps,
                        .greedy_training_episode_interval =
                            options
                                .reinforcement_greedy_training_episode_interval,
                        .greedy_training_epsilon =
                            options.reinforcement_greedy_training_epsilon,
                        .random_seed =
                            options.random_seed.value_or(0x524c444eU),
                        .character = player_character,
                        .opponent_character = opponent_character,
                        .facing = facing,
                        .training_enabled =
                            options.reinforcement_training_enabled,
                        .checkpoint_writes_enabled =
                            options.reinforcement_training_enabled,
                        .checkpoint_interval_episodes =
                            options
                                .reinforcement_checkpoint_interval_episodes,
                        .resume = effective_resume,
                        .save_replay = true,
                        .recurrent_backend =
                            options.reinforcement_backend,
                        .trainable_lstm =
                            options.reinforcement_trainable_lstm,
                        .trainable_resnet =
                            options.reinforcement_trainable_resnet,
                        .temporal_visual_size =
                            static_cast<std::size_t>(
                                ResNet50FeatureExtractor::kFeatureSize
                            ),
                        .temporal_projection_size =
                            options.temporal_visual_projection_size,
                        .temporal_controller_size =
                            TemporalLstmConfig::kControllerFeatureCount,
                        .temporal_hidden_size =
                            options.temporal_hidden_size,
                        .temporal_sequence_capacity =
                            options.observation_frames,
                        .temporal_device_id =
                            options.temporal_device_id,
                        .resnet50_checkpoint_path =
                            options.resnet50_checkpoint,
                        .resnet_learning_rate =
                            options
                                .reinforcement_resnet_learning_rate,
                        .resnet_batch_size =
                            options.reinforcement_resnet_batch_size,
                        .resnet_update_interval =
                            options
                                .reinforcement_resnet_update_interval,
                        .resnet_replay_images =
                            options
                                .reinforcement_resnet_replay_images,
                        .recurrent_service_script =
                            options.recurrent_service_script,
                        .recurrent_checkpoint_path =
                            recurrent_checkpoint,
                        .recurrent_migration_checkpoint_path =
                            options.reinforcement_migration_checkpoint,
                        .checkpoint_path = latest_checkpoint,
                        .best_checkpoint_path = best_checkpoint,
                    }
                );
            } else {
                reinforcement_agent_storage->configure_session(
                    player_character,
                    opponent_character,
                    facing,
                    best_checkpoint
                );
                reinforcement_agent_reused = true;
            }
        } catch (const std::exception& error) {
            writer.send_json({
                {"protocol_version", kActionProtocolVersion},
                {"type", "session_config_error"},
                {"message",
                 std::string("could not initialize character agent: ") +
                     error.what()},
            });
            send_close(writer, 1011, "character agent initialization failed");
            return;
        }
    }
    ReinforcementAgent* reinforcement_agent =
        reinforcement_agent_storage.get();
    MockAgent agent(options.random_seed, facing, player_character);

    if (explicit_session_configuration) {
        if (!writer.send_json({
                {"protocol_version", kActionProtocolVersion},
                {"type", v21_session
                    ? "configure_session_ack"
                    : "session_config_ack"},
                {"request_id", configure_session_metadata.request_id},
                {"timestamp_ms", epoch_milliseconds()},
                {"player_character",
                 fighter_character_slug(player_character)},
                {"player_character_name",
                 fighter_character_name(player_character)},
                {"opponent_character",
                 opponent_character_slug(opponent_character)},
                {"opponent_character_name",
                 opponent_character_name(opponent_character)},
                {"opponent_feature_index",
                 static_cast<std::size_t>(opponent_character)},
                {"opponent_embedding_index",
                 static_cast<std::size_t>(opponent_character)},
                {"opponent_feature_count",
                 kOpponentCharacterFeatureCount},
                {"opponent_vocabulary_size",
                 kOpponentCharacterFeatureCount},
                {"opponent_embedding_size", 16U},
                {"architecture_version", kServerArchitectureVersion},
                {"encounter_type",
                 encounter_type_slug(encounter_configuration.type)},
                {"encounter_stage",
                 encounter_configuration.current_stage},
                {"opponent_count",
                 encounter_configuration.opponent_count},
                {"remaining_opponents",
                 encounter_configuration.opponent_count -
                    encounter_configuration.current_stage - 1U},
                {"player_side", player_is_left ? "left" : "right"},
                {"initial_facing",
                 facing == FighterFacing::right ? "right" : "left"},
                {"profile",
                 std::string(fighter_character_slug(player_character)) +
                    "_v1"},
                {"mock_move_count",
                 make_character_curriculum(
                     player_character, facing
                 ).size()},
                {"rl_option_count",
                 reinforcement_agent != nullptr
                     ? reinforcement_agent->option_names().size()
                     : 0U},
                {"agent_mode",
                 reinforcement_agent != nullptr
                     ? "reinforcement_learning"
                     : "mock_teacher"},
                {"shared_policy",
                 reinforcement_agent != nullptr},
                {"rl_replay_retained_across_connections",
                 reinforcement_agent != nullptr},
                {"rl_replay_size",
                 reinforcement_agent != nullptr
                     ? reinforcement_agent->stats().replay_size
                     : 0U},
                {"rl_training_enabled",
                 reinforcement_agent != nullptr &&
                     options.reinforcement_training_enabled},
                {"rl_read_only",
                 reinforcement_agent != nullptr &&
                     !options.reinforcement_training_enabled},
                {"rl_epsilon_start",
                 options.reinforcement_epsilon_start},
                {"rl_epsilon_final",
                 options.reinforcement_epsilon_final},
                {"rl_gamma", options.reinforcement_gamma},
                {"rl_q_value_limit",
                 options.reinforcement_q_value_limit},
                {"rl_terminal_replay_fraction",
                 options.reinforcement_terminal_replay_fraction},
                {"rl_hard_negative_replay_fraction",
                 options.reinforcement_hard_negative_replay_fraction},
                {"rl_n_step_options",
                 options.reinforcement_n_step_options},
                {"rl_conservative_bootstrap",
                 options.reinforcement_conservative_bootstrap},
                {"rl_twin_critics", true},
                {"rl_checkpoint_interval_episodes",
                 options
                     .reinforcement_checkpoint_interval_episodes},
                {"timed_sequences_enabled",
                 options.timed_sequences_enabled},
                {"timed_sequence_version", 2},
                {"timed_sequence_step_semantics",
                 "exact_controller_state"},
                {"supported_action_modes", {
                    "state", "timed_sequence"
                }},
                {"sequence_input_hold_ms",
                 options.sequence_input_hold.count()},
                {"sequence_neutral_hold_ms",
                 options.sequence_neutral_hold.count()},
                {"sequence_final_release_ms",
                 options.sequence_final_release.count()},
                {"sequence_actionable_ticks",
                 options.sequence_actionable_ticks},
                {"training_reset_state_enabled",
                 reinforcement_agent != nullptr &&
                     options.training_reset_state_enabled},
                {"training_reset_non_fight_timeout_ms",
                 options.training_reset_non_fight_timeout.count()},
                {"training_reset_round_intro_timeout_ms",
                 options.training_reset_round_intro_timeout.count()},
                {"frame_stream_stall_timeout_ms",
                 options.frame_stream_stall_timeout.count()},
                {"frame_stream_recovery_interval_ms",
                 options.frame_stream_recovery_interval.count()},
                {"repeated_frame_pause_ms",
                 options.repeated_frame_pause_timeout.count()},
                {"repeated_frame_reset_ms",
                 options.repeated_frame_reset_timeout.count()},
                {"training_reset_trigger",
                 options.reinforcement_training_enabled
                     ? "round_lost_or_non_fight_timeout"
                     : "round_terminal_or_non_fight_timeout"},
                {"training_reset_ack_type", "training_reset_ack"},
                {"checkpoint_loaded",
                 reinforcement_agent != nullptr &&
                     reinforcement_agent->stats().checkpoint_loaded},
                {"trainable_lstm",
                 reinforcement_agent != nullptr &&
                     reinforcement_agent->stats().trainable_lstm},
                {"recurrent_backend",
                 options.reinforcement_backend},
                {"recurrent_checkpoint_loaded",
                 reinforcement_agent != nullptr &&
                     reinforcement_agent
                         ->stats()
                         .recurrent_checkpoint_loaded},
                {"trainable_resnet",
                 reinforcement_agent != nullptr &&
                     reinforcement_agent->stats().trainable_resnet},
                {"resnet_checkpoint_loaded",
                 reinforcement_agent != nullptr &&
                     reinforcement_agent
                         ->stats()
                         .resnet_checkpoint_loaded},
                {"combat_state_version",
                 reinforcement_agent != nullptr ? 2U : 0U},
                {"combat_context_features",
                 reinforcement_agent != nullptr
                     ? kReinforcementContextFeatureCount
                     : 0U},
                {"damage_dealt_reward_scale",
                 reinforcement_agent != nullptr
                     ? options
                           .reinforcement_damage_dealt_reward_scale
                     : 0.0f},
                {"damage_taken_penalty_scale",
                 reinforcement_agent != nullptr
                     ? options
                           .reinforcement_damage_taken_penalty_scale
                     : 0.0f},
                {"round_win_bonus",
                 reinforcement_agent != nullptr
                     ? options.reinforcement_round_win_bonus
                     : 0.0f},
                {"round_loss_penalty",
                 reinforcement_agent != nullptr
                     ? options.reinforcement_round_loss_penalty
                     : 0.0f},
                {"combo_reward_scale",
                 reinforcement_agent != nullptr
                     ? options
                           .reinforcement_combo_reward_scale
                     : 0.0f},
                {"defense_reward_per_second",
                 reinforcement_agent != nullptr
                     ? options
                           .reinforcement_defense_reward_per_second
                     : 0.0f},
                {"idle_penalty_per_second",
                 reinforcement_agent != nullptr
                     ? options
                           .reinforcement_idle_penalty_per_second
                     : 0.0f},
                {"corner_penalty_per_second",
                 reinforcement_agent != nullptr
                     ? options
                           .reinforcement_corner_penalty_per_second
                     : 0.0f},
                {"neutral_option_penalty",
                 reinforcement_agent != nullptr
                     ? options
                           .reinforcement_neutral_option_penalty
                     : 0.0f},
                {"rl_training_device",
                 reinforcement_agent != nullptr
                     ? reinforcement_agent->stats().training_device
                     : "none"},
                {"rl_checkpoint",
                 reinforcement_agent != nullptr
                     ? reinforcement_agent
                           ->config()
                           .checkpoint_path
                           .string()
                     : ""},
            })) {
            return;
        }
    }
    metrics.write({
        {"event", "session_config"},
        {"connection_id", connection_id},
        {"timestamp_ms", epoch_milliseconds()},
        {"explicit", explicit_session_configuration},
        {"player_character", fighter_character_slug(player_character)},
        {"opponent_character",
         opponent_character_slug(opponent_character)},
        {"player_side", player_is_left ? "left" : "right"},
        {"initial_facing",
         facing == FighterFacing::right ? "right" : "left"},
        {"agent_mode",
         reinforcement_agent != nullptr
             ? "reinforcement_learning"
             : "mock_teacher"},
        {"rl_training_enabled",
         reinforcement_agent != nullptr &&
             options.reinforcement_training_enabled},
        {"rl_read_only",
         reinforcement_agent != nullptr &&
             !options.reinforcement_training_enabled},
        {"rl_agent_reused", reinforcement_agent_reused},
        {"rl_replay_retained_across_connections",
         reinforcement_agent != nullptr},
        {"rl_replay_size",
         reinforcement_agent != nullptr
             ? reinforcement_agent->stats().replay_size
             : 0U},
        {"rl_checkpoint_interval_episodes",
         options.reinforcement_checkpoint_interval_episodes},
        {"damage_dealt_reward_scale",
         options.reinforcement_damage_dealt_reward_scale},
        {"damage_taken_penalty_scale",
         options.reinforcement_damage_taken_penalty_scale},
        {"round_win_bonus",
         options.reinforcement_round_win_bonus},
        {"round_loss_penalty",
         options.reinforcement_round_loss_penalty},
        {"combo_reward_scale",
         options.reinforcement_combo_reward_scale},
        {"defense_reward_per_second",
         options.reinforcement_defense_reward_per_second},
        {"idle_penalty_per_second",
         options.reinforcement_idle_penalty_per_second},
        {"corner_penalty_per_second",
         options.reinforcement_corner_penalty_per_second},
        {"neutral_option_penalty",
         options.reinforcement_neutral_option_penalty},
        {"neutral_suppressed_during_fight",
         options.reinforcement_suppress_neutral_during_fight},
        {"combo_max_distance",
         options.reinforcement_combo_max_distance},
        {"close_attack_max_distance",
         options.reinforcement_close_attack_max_distance},
        {"throw_max_distance", options.reinforcement_throw_max_distance},
        {"sweep_max_distance", options.reinforcement_sweep_max_distance},
        {"projectile_preferred_min_distance",
         options.reinforcement_projectile_preferred_min_distance},
        {"projectile_q_preference_margin",
         options.reinforcement_projectile_q_preference_margin},
        {"disable_cloud_min_distance",
         options.reinforcement_disable_cloud_min_distance},
        {"disable_cloud_max_distance",
         options.reinforcement_disable_cloud_max_distance},
        {"advance_min_distance",
         options.reinforcement_advance_min_distance},
        {"defense_max_distance",
         options.reinforcement_defense_max_distance},
        {"dynamic_facing", options.reinforcement_dynamic_facing},
        {"disable_cloud_cooldown_options",
         options.reinforcement_disable_cloud_cooldown_options},
        {"throw_cooldown_options",
         options.reinforcement_throw_cooldown_options},
        {"sweep_cooldown_options",
         options.reinforcement_sweep_cooldown_options},
        {"melee_impact_credit_steps",
         options.reinforcement_melee_impact_credit_steps},
        {"teleport_impact_credit_steps",
         options.reinforcement_teleport_impact_credit_steps},
        {"projectile_impact_credit_min_steps",
         options.reinforcement_projectile_impact_credit_min_steps},
        {"projectile_impact_credit_max_steps",
         options.reinforcement_projectile_impact_credit_max_steps},
        {"greedy_training_episode_interval",
         options.reinforcement_greedy_training_episode_interval},
        {"greedy_training_epsilon",
         options.reinforcement_greedy_training_epsilon},
        {"trainable_lstm",
         reinforcement_agent != nullptr &&
             reinforcement_agent->stats().trainable_lstm},
        {"recurrent_backend", options.reinforcement_backend},
        {"trainable_resnet",
         reinforcement_agent != nullptr &&
             reinforcement_agent->stats().trainable_resnet},
        {"rl_training_device",
         reinforcement_agent != nullptr
             ? reinforcement_agent->stats().training_device
             : "none"},
        {"training_reset_state_enabled",
         reinforcement_agent != nullptr &&
             options.training_reset_state_enabled},
        {"training_reset_non_fight_timeout_ms",
         options.training_reset_non_fight_timeout.count()},
        {"training_reset_round_intro_timeout_ms",
         options.training_reset_round_intro_timeout.count()},
        {"frame_stream_stall_timeout_ms",
         options.frame_stream_stall_timeout.count()},
        {"frame_stream_recovery_interval_ms",
         options.frame_stream_recovery_interval.count()},
        {"repeated_frame_pause_ms",
         options.repeated_frame_pause_timeout.count()},
        {"repeated_frame_reset_ms",
         options.repeated_frame_reset_timeout.count()},
        {"rl_checkpoint",
         reinforcement_agent != nullptr
             ? reinforcement_agent->config().checkpoint_path.string()
             : ""},
    });
    std::cout
        << "[SESSION] connection=" << connection_id
        << " character=" << fighter_character_slug(player_character)
        << " opponent=" << opponent_character_slug(opponent_character)
        << " side=" << (player_is_left ? "left" : "right")
        << " facing="
        << (facing == FighterFacing::right ? "right" : "left")
        << " mode="
        << (
            reinforcement_agent != nullptr
                ? "reinforcement_learning"
                : "mock_teacher"
        )
        << " checkpoint="
        << (
            reinforcement_agent != nullptr
                ? reinforcement_agent->config().checkpoint_path.string()
                : "none"
        )
        << " recurrent="
        << (
            reinforcement_agent != nullptr &&
                    reinforcement_agent->stats().trainable_lstm
                ? reinforcement_agent->stats().training_device
                : "disabled"
        )
        << " backend=" << options.reinforcement_backend
        << " replay="
        << (
            reinforcement_agent != nullptr
                ? reinforcement_agent->stats().replay_size
                : 0U
        )
        << " reused=" << (reinforcement_agent_reused ? 1 : 0)
        << " resnet="
        << (
            reinforcement_agent != nullptr &&
                    reinforcement_agent->stats().trainable_resnet
                ? "trainable_stage4"
                : "frozen"
        )
        << '\n';

    CircularFrameBuffer buffer(options.frame_buffer_capacity);
    std::unique_ptr<RepeatedFrameDetector> repeated_frame_detector;
    std::unique_ptr<FightStateTracker> fight_tracker;
    std::unique_ptr<CombatStateTracker> combat_tracker;
    std::unique_ptr<TrainingResetTracker> training_reset_tracker;
    if (reinforcement_agent != nullptr) {
        repeated_frame_detector =
            std::make_unique<RepeatedFrameDetector>(
                RepeatedFrameDetectorConfig{
                    .pause_after =
                        options.repeated_frame_pause_timeout,
                    .reset_after =
                        options.repeated_frame_reset_timeout,
                }
            );
        fight_tracker = std::make_unique<FightStateTracker>(FightStateConfig{
            .player_side = player_is_left
                ? PlayerSide::left
                : PlayerSide::right,
            .maximum_action_frame_age_ms =
                std::max<std::int64_t>(
                    250,
                    options.decision_interval.count() * 3
                ),
            .damage_dealt_reward_scale =
                options.reinforcement_damage_dealt_reward_scale,
            .damage_taken_penalty_scale =
                options.reinforcement_damage_taken_penalty_scale,
            .round_win_bonus =
                options.reinforcement_round_win_bonus,
            .round_loss_penalty =
                options.reinforcement_round_loss_penalty,
        });
        fight_tracker->set_external_encounter_lifecycle(
            v21_session && requires_external_encounter_lifecycle(
                encounter_configuration
            )
        );
        combat_tracker = std::make_unique<CombatStateTracker>(
            CombatStateConfig{
                .player_side = player_is_left
                    ? PlayerSide::left
                    : PlayerSide::right,
                .combo_reward_scale =
                    options.reinforcement_combo_reward_scale,
                .defense_reward_per_second =
                    options
                        .reinforcement_defense_reward_per_second,
                .idle_penalty_per_second =
                    options
                        .reinforcement_idle_penalty_per_second,
                .corner_penalty_per_second =
                    options
                        .reinforcement_corner_penalty_per_second,
            }
        );
        if (options.training_reset_state_enabled) {
            training_reset_tracker =
                std::make_unique<TrainingResetTracker>(
                    !options.reinforcement_training_enabled,
                    options.training_reset_non_fight_timeout.count(),
                    options.training_reset_round_intro_timeout.count()
                );
        }
        reinforcement_agent->reset_episode(false);
    }
    if (temporal_agent != nullptr) {
        temporal_agent->reset();
    }
    ActionHistory action_history(
        std::max<std::size_t>(64U, options.applied_action_history_size * 8U)
    );
    std::atomic_bool connected{true};
    std::atomic<std::uint64_t> received_frames{0};
    std::atomic<std::uint64_t> decode_errors{0};
    std::atomic<std::uint64_t> acknowledgements_received{0};
    std::atomic<std::uint64_t> acknowledgements_rejected{0};
    std::atomic<std::uint64_t> training_reset_requests{0};
    std::atomic<std::uint64_t> training_reset_retries{0};
    std::atomic<std::uint64_t> training_reset_acknowledgements{0};
    std::atomic<std::uint64_t>
        training_reset_acknowledgements_rejected{0};
    std::atomic<std::uint64_t> last_training_reset_id{0};
    std::atomic<std::uint64_t> last_training_reset_ack_id{0};
    std::atomic<std::uint64_t> client_message_errors{0};
    std::atomic<std::uint64_t> temporal_model_errors{0};
    std::atomic<std::uint64_t> reinforcement_errors{0};
    std::atomic<std::uint64_t> temporal_frames_overwritten{0};
    std::atomic<std::uint64_t> temporal_frames_processed{0};
    std::atomic<std::uint64_t> persistence_frames_dropped{0};
    std::atomic<std::uint64_t> persistence_frames_saved{0};
    std::atomic<std::uint64_t> persistence_errors{0};
    std::atomic_bool repeated_frame_training_paused{false};
    std::atomic<std::int64_t> repeated_frame_duration_ms{0};
    std::atomic<std::uint64_t> repeated_frame_count{0};
    std::atomic<std::uint64_t> repeated_frame_reset_generation{0};
    std::atomic<std::uint64_t> repeated_frame_reset_frame_id{0};
    Clock::time_point next_debug_capture = Clock::time_point::min();
    std::atomic_bool workers_stopping{false};
    std::mutex control_queue_mutex;
    std::deque<ClientTextMessage> control_queue;
    const auto enqueue_control_message = [&](ClientTextMessage message) {
        std::lock_guard lock(control_queue_mutex);
        control_queue.push_back(std::move(message));
    };

    std::mutex temporal_queue_mutex;
    std::condition_variable temporal_queue_changed;
    std::shared_ptr<const BufferedFrame> pending_temporal_frame;
    const auto enqueue_temporal =
        [&](std::shared_ptr<const BufferedFrame> frame) {
            if (temporal_agent == nullptr) {
                return;
            }
            {
                std::lock_guard lock(temporal_queue_mutex);
                if (pending_temporal_frame != nullptr) {
                    temporal_frames_overwritten.fetch_add(1);
                }
                pending_temporal_frame = std::move(frame);
            }
            temporal_queue_changed.notify_one();
        };
    std::jthread temporal_worker([&](std::stop_token) {
        while (true) {
            std::shared_ptr<const BufferedFrame> frame;
            {
                std::unique_lock lock(temporal_queue_mutex);
                temporal_queue_changed.wait(lock, [&] {
                    return workers_stopping.load() ||
                        pending_temporal_frame != nullptr;
                });
                if (pending_temporal_frame == nullptr &&
                    workers_stopping.load()) {
                    break;
                }
                frame = std::exchange(
                    pending_temporal_frame, nullptr
                );
            }
            try {
                const auto result = temporal_agent->observe_frame(*frame);
                temporal_frames_processed.fetch_add(1);
                metrics.write({
                    {"event", "temporal_observation"},
                    {"connection_id", connection_id},
                    {"frame_id", result.frame_id},
                    {"sequence_size", result.sequence_size},
                    {"feature_ms", result.feature_ms},
                    {"frames_overwritten",
                     temporal_frames_overwritten.load()},
                });
            } catch (const std::exception& error) {
                temporal_model_errors.fetch_add(1);
                std::cerr << "[TEMPORAL] connection=" << connection_id
                          << " frame=" << frame->frame_id
                          << " observation failed: " << error.what() << '\n';
            }
        }
    });

    std::mutex persistence_queue_mutex;
    std::condition_variable persistence_queue_changed;
    std::deque<FramePersistenceJob> persistence_queue;
    const auto enqueue_persistence =
        [&](std::shared_ptr<const FrameEnvelope> envelope,
            std::int64_t received_timestamp_ms) {
            const bool training_capture =
                options.collect_training_data;
            const auto persistence_now = Clock::now();
            const bool debug_capture =
                options.save_frames &&
                persistence_now >= next_debug_capture;
            if (!training_capture && !debug_capture) {
                return;
            }
            if (debug_capture) {
                next_debug_capture =
                    persistence_now + options.debug_capture_interval;
            }
            FramePersistenceJob job{
                .envelope = std::move(envelope),
                .received_timestamp_ms = received_timestamp_ms,
                .latest_applied_action =
                    latest_applied_record(action_history),
                .player_character =
                    std::string(
                        fighter_character_slug(player_character)
                    ),
                .opponent_character =
                    std::string(
                        opponent_character_slug(
                            active_opponent_character.load()
                        )
                    ),
                .training_capture = training_capture,
            };
            {
                std::lock_guard lock(persistence_queue_mutex);
                if (persistence_queue.size() ==
                    options.persistence_queue_capacity) {
                    persistence_queue.pop_front();
                    persistence_frames_dropped.fetch_add(1);
                }
                persistence_queue.push_back(std::move(job));
            }
            persistence_queue_changed.notify_one();
        };
    std::jthread persistence_worker([&](std::stop_token) {
        while (true) {
            FramePersistenceJob job;
            {
                std::unique_lock lock(persistence_queue_mutex);
                persistence_queue_changed.wait(lock, [&] {
                    return workers_stopping.load() ||
                        !persistence_queue.empty();
                });
                if (persistence_queue.empty() &&
                    workers_stopping.load()) {
                    break;
                }
                job = std::move(persistence_queue.front());
                persistence_queue.pop_front();
            }
            try {
                save_frame(options, connection_id, job);
                persistence_frames_saved.fetch_add(1);
            } catch (const std::exception& error) {
                persistence_errors.fetch_add(1);
                std::cerr << "[CAPTURE] connection=" << connection_id
                          << " frame=" << job.envelope->frame_id
                          << " save failed: " << error.what() << '\n';
            }
        }
    });

    std::jthread inference_thread([&](std::stop_token thread_stop) {
        std::uint64_t decision_id = 0;
        auto next_decision = Clock::now() + options.decision_interval;
        std::optional<nlohmann::json> pending_training_reset_request;
        std::optional<std::string>
            pending_training_reset_detected_reason;
        std::size_t pending_training_reset_attempt = 0;
        std::optional<ResetReason> pending_terminal_reset_reason;
        std::uint64_t telemetry_request_id = 1;
        std::uint64_t evaluation_session_id =
            options.reinforcement_training_enabled ? 0U : 1U;
        EncounterEvent latest_encounter_event = EncounterEvent::none;
        bool opponent_health_delta_ignored = false;
        float latest_stage_bonus = 0.0f;
        auto next_telemetry_sample = Clock::time_point::min();
        EpisodeStatistics episode_statistics;
        std::map<std::string, ModeStatistics> mode_statistics;
        ResetStatistics reset_statistics;
        std::size_t consecutive_reset_failures = 0;
        const auto statistics_key = [&](AgentMode mode) {
            std::ostringstream key;
            key << agent_mode_slug(mode) << ':'
                << fighter_character_slug(player_character) << ':'
                << encounter_type_slug(encounter_configuration.type) << ':';
            for (std::size_t index = 0;
                 index < encounter_configuration.opponent_count;
                 ++index) {
                if (index != 0) {
                    key << ',';
                }
                key << opponent_character_slug(
                    encounter_configuration.opponents[index]
                );
            }
            return key.str();
        };
        const auto send_telemetry_marker = [&](std::string marker_type,
                                                std::uint64_t episode_id,
                                                float value,
                                                std::string label) {
            nlohmann::json marker{
                {"protocol_version", kServerProtocolVersion},
                {"request_id", telemetry_request_id++},
                {"timestamp_ms", epoch_milliseconds()},
                {"type", "telemetry_marker"},
                {"episode_id", episode_id},
                {"marker_type", std::move(marker_type)},
                {"value", value},
                {"label", std::move(label)},
            };
            writer.send_json(marker);
            marker["event"] = "telemetry_marker";
            marker["connection_id"] = connection_id;
            metrics.write(marker);
        };
        const auto require_client_reconnect = [&] (
            std::uint64_t reset_id,
            std::uint64_t episode_id,
            std::string_view failure_reason
        ) {
            const std::int64_t timestamp_ms = epoch_milliseconds();
            nlohmann::json message{
                {"protocol_version", kServerProtocolVersion},
                {"request_id", telemetry_request_id++},
                {"timestamp_ms", timestamp_ms},
                {"type", "reconnect_required"},
                {"reason", "reset_recovery_exhausted"},
                {"failure_reason", failure_reason},
                {"reset_id", reset_id},
                {"episode_id", episode_id},
                {"reconnect_after_ms", 1000},
                {"release_controls", true},
                {"discard_inflight_action", true},
                {"must_reconfigure_session", true},
            };
            writer.send_json(message);
            message["event"] = "reconnect_required";
            message["connection_id"] = connection_id;
            message["consecutive_reset_failures"] =
                consecutive_reset_failures;
            metrics.write(message);
            std::cerr
                << "[TRAINING_RESET] connection=" << connection_id
                << " reset_id=" << reset_id
                << " recovery exhausted after "
                << consecutive_reset_failures
                << " failures; forcing WebSocket reconnect\n";
            send_close(
                writer,
                1012,
                "reset recovery requires reconnect"
            );
            connected.store(false);
            shutdown(socket, SHUT_RDWR);
        };
        std::uint64_t handled_repeated_frame_reset_generation = 0;
        bool inference_repeated_frame_paused = false;
        auto next_training_reset_retry = Clock::time_point::min();
        const auto training_reset_retry_interval =
            std::chrono::milliseconds(500);
        bool frame_stream_recovery_active = false;
        std::uint64_t frame_stream_recovery_id = 0;
        std::uint64_t frame_stream_recovery_last_frame_id = 0;
        std::size_t frame_stream_recovery_attempt = 0;
        std::int64_t frame_stream_recovery_started_ms = 0;
        auto next_frame_stream_recovery = Clock::time_point::min();
        auto next_reward_log =
            Clock::now() + options.reinforcement_reward_log_interval;
        float reward_log_accumulator = 0.0f;
        float terminal_reward_log_accumulator = 0.0f;
        float damage_dealt_log_accumulator = 0.0f;
        float damage_taken_log_accumulator = 0.0f;
        float shaping_reward_log_accumulator = 0.0f;
        float combo_reward_log_accumulator = 0.0f;
        float defense_reward_log_accumulator = 0.0f;
        float idle_penalty_log_accumulator = 0.0f;
        float corner_penalty_log_accumulator = 0.0f;
        float neutral_penalty_log_accumulator = 0.0f;
        std::uint64_t reward_log_decisions = 0;
        std::uint64_t reward_log_active_decisions = 0;
        std::size_t player_actionable_ticks = 0;
        while (!thread_stop.stop_requested() && connected.load() && !stop_requested.load()) {
            std::this_thread::sleep_until(next_decision);
            if (thread_stop.stop_requested() || !connected.load() || stop_requested.load()) {
                break;
            }

            const auto inference_start = Clock::now();
            const auto observation = buffer.latest(options.observation_frames);
            if (!observation.empty()) {
                const std::int64_t decision_tick_ms =
                    epoch_milliseconds();
                const std::uint64_t current_decision_id = ++decision_id;
                const auto applied_actions = action_history.latest_applied(
                    options.applied_action_history_size
                );
                const auto latest_applied =
                    latest_applied_record(action_history);
                latest_encounter_event = EncounterEvent::none;
                opponent_health_delta_ignored = false;
                latest_stage_bonus = 0.0f;
                bool stage_completed_this_tick = false;
                bool reset_reconnect_required = false;
                std::uint64_t reconnect_reset_id = 0;
                std::uint64_t reconnect_episode_id = 0;
                std::string reconnect_failure_reason;
                std::deque<ClientTextMessage> control_messages;
                {
                    std::lock_guard lock(control_queue_mutex);
                    control_messages.swap(control_queue);
                }
                for (ClientTextMessage& control_message : control_messages) {
                    if (std::holds_alternative<
                            UpdateEncounterStageRequest
                        >(control_message)) {
                        const auto& request = std::get<
                            UpdateEncounterStageRequest
                        >(control_message);
                        EncounterTransition transition;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            transition = protocol_session.apply_stage_update(
                                request
                            );
                        }
                        if (transition.accepted && !transition.duplicate) {
                            fight_tracker->advance_encounter_stage(
                                transition.stage_bonus
                            );
                            active_opponent_character.store(
                                transition.current_opponent
                            );
                            encounter_configuration.current_stage =
                                request.current_stage;
                            fight_tracker->set_external_encounter_lifecycle(
                                requires_external_encounter_lifecycle(
                                    encounter_configuration
                                )
                            );
                            reinforcement_agent->configure_encounter_stage(
                                transition.current_opponent
                            );
                            latest_encounter_event =
                                EncounterEvent::stage_completed;
                            stage_completed_this_tick = true;
                            latest_stage_bonus = transition.stage_bonus;
                            opponent_health_delta_ignored = true;
                            std::cout
                                << "\n[RL][ENCOUNTER] episode="
                                << request.episode_id
                                << " type=endurance stage="
                                << static_cast<unsigned>(
                                       request.current_stage + 1U
                                   )
                                << '/'
                                << static_cast<unsigned>(
                                       encounter_configuration.opponent_count
                                   )
                                << " previous="
                                << opponent_character_slug(
                                       transition.previous_opponent
                                   )
                                << " current="
                                << opponent_character_slug(
                                       transition.current_opponent
                                   )
                                << " health_delta_skipped=true stage_bonus="
                                << transition.stage_bonus << "\n\n"
                                << std::flush;
                            send_telemetry_marker(
                                "stage_completed",
                                request.episode_id,
                                transition.stage_bonus,
                                std::string(opponent_character_slug(
                                    transition.current_opponent
                                ))
                            );
                        }
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", request.metadata.request_id},
                            {"timestamp_ms", epoch_milliseconds()},
                            {"type", transition.accepted
                                ? "encounter_stage_changed"
                                : "error"},
                            {"accepted", transition.accepted},
                            {"duplicate", transition.duplicate},
                            {"episode_id", request.episode_id},
                            {"stage_transition_id",
                             request.stage_transition_id},
                            {"encounter_stage",
                             request.current_stage},
                            {"current_opponent",
                             opponent_character_slug(
                                 transition.current_opponent
                             )},
                            {"stage_bonus", transition.stage_bonus},
                            {"opponent_health_delta_ignored",
                             transition.opponent_health_delta_ignored},
                            {"message", transition.message},
                        });
                        continue;
                    }
                    if (std::holds_alternative<EncounterEventRequest>(
                            control_message
                        )) {
                        const auto& request =
                            std::get<EncounterEventRequest>(control_message);
                        EncounterTransition transition;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            transition = protocol_session.apply_encounter_event(
                                request
                            );
                        }
                        if (transition.accepted && !transition.duplicate) {
                            latest_encounter_event = request.event;
                            if (request.event ==
                                EncounterEvent::stage_completed) {
                                fight_tracker->advance_encounter_stage(
                                    transition.stage_bonus
                                );
                                active_opponent_character.store(
                                    transition.current_opponent
                                );
                                encounter_configuration.current_stage =
                                    static_cast<std::uint8_t>(
                                        request.encounter_stage + 1U
                                    );
                                fight_tracker
                                    ->set_external_encounter_lifecycle(
                                        requires_external_encounter_lifecycle(
                                            encounter_configuration
                                        )
                                    );
                                reinforcement_agent
                                    ->configure_encounter_stage(
                                        transition.current_opponent
                                    );
                                latest_stage_bonus = transition.stage_bonus;
                                opponent_health_delta_ignored = true;
                                stage_completed_this_tick = true;
                            } else if (transition.terminal) {
                                fight_tracker->finish_encounter(
                                    transition.won
                                );
                                pending_terminal_reset_reason =
                                    transition.won
                                        ? ResetReason::encounter_completed
                                        : ResetReason::player_defeated;
                            }
                            send_telemetry_marker(
                                std::string(encounter_event_slug(request.event)),
                                request.episode_id,
                                transition.stage_bonus,
                                std::string(opponent_character_slug(
                                    transition.current_opponent
                                ))
                            );
                        }
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", request.metadata.request_id},
                            {"timestamp_ms", epoch_milliseconds()},
                            {"type", transition.accepted
                                ? "encounter_event_ack"
                                : "error"},
                            {"accepted", transition.accepted},
                            {"duplicate", transition.duplicate},
                            {"episode_id", request.episode_id},
                            {"event", encounter_event_slug(request.event)},
                            {"terminal", transition.terminal},
                            {"stage_bonus", transition.stage_bonus},
                            {"message", transition.message},
                        });
                        continue;
                    }
                    if (std::holds_alternative<SetAgentModeRequest>(
                            control_message
                        )) {
                        const auto& request =
                            std::get<SetAgentModeRequest>(control_message);
                        ModeChangeResult result;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            result = protocol_session.request_agent_mode(
                                request
                            );
                        }
                        if (!result.pending) {
                            reinforcement_agent->set_agent_mode(
                                result.effective_mode
                            );
                            if (result.effective_mode ==
                                    AgentMode::evaluation &&
                                request.reset_evaluation_session) {
                                ++evaluation_session_id;
                            }
                        }
                        if (result.changed || result.pending) {
                            send_telemetry_marker(
                                "mode_changed",
                                episode_statistics.episode_id,
                                result.effective_mode == AgentMode::evaluation
                                    ? 0.0f
                                    : reinforcement_agent->stats().epsilon,
                                result.pending
                                    ? std::string("pending:") +
                                        std::string(agent_mode_slug(request.mode))
                                    : std::string(agent_mode_slug(
                                        result.effective_mode
                                    ))
                            );
                        }
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", request.metadata.request_id},
                            {"timestamp_ms", epoch_milliseconds()},
                            {"type", "agent_mode_changed"},
                            {"effective_mode",
                             agent_mode_slug(result.effective_mode)},
                            {"pending", result.pending},
                            {"pending_mode", result.pending_mode.has_value()
                                ? agent_mode_slug(*result.pending_mode)
                                : "none"},
                            {"effective_epsilon",
                             result.effective_mode == AgentMode::evaluation
                                ? 0.0f
                                : reinforcement_agent->stats().epsilon},
                            {"weights_frozen",
                             result.effective_mode == AgentMode::evaluation},
                            {"replay_frozen",
                             result.effective_mode == AgentMode::evaluation},
                            {"scheduler_frozen",
                             result.effective_mode == AgentMode::evaluation},
                        });
                        continue;
                    }
                    if (std::holds_alternative<SetExplorationRequest>(
                            control_message
                        )) {
                        const auto& request =
                            std::get<SetExplorationRequest>(control_message);
                        ExplorationChangeResult result;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            result = protocol_session.set_exploration(request);
                        }
                        if (result.accepted) {
                            reinforcement_agent->set_exploration(
                                result.mode,
                                result.value,
                                result.matchup_counter_reset
                            );
                            send_telemetry_marker(
                                "epsilon_changed",
                                episode_statistics.episode_id,
                                reinforcement_agent->stats().epsilon,
                                std::string(exploration_mode_slug(result.mode))
                            );
                        }
                        const bool evaluation = [&] {
                            std::lock_guard lock(protocol_session_mutex);
                            return protocol_session.mode() ==
                                AgentMode::evaluation;
                        }();
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", request.metadata.request_id},
                            {"timestamp_ms", epoch_milliseconds()},
                            {"type", result.accepted
                                ? "exploration_changed"
                                : "error"},
                            {"exploration_mode",
                             exploration_mode_slug(result.mode)},
                            {"configured_value", result.value},
                            {"effective_epsilon", evaluation
                                ? 0.0f
                                : reinforcement_agent->stats().epsilon},
                            {"reset_matchup_counter",
                             result.matchup_counter_reset},
                        });
                        continue;
                    }
                    if (std::holds_alternative<RequestReset>(
                            control_message
                        )) {
                        const auto& request =
                            std::get<RequestReset>(control_message);
                        const std::uint64_t reset_id =
                            last_training_reset_id.fetch_add(1U) + 1U;
                        ResetTransition result;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            result = protocol_session.request_reset(
                                request.reason,
                                request.episode_id,
                                reset_id,
                                epoch_milliseconds(),
                                false
                            );
                        }
                        if (result.accepted) {
                            reinforcement_agent->suspend();
                            ++reset_statistics.resets_requested;
                            if (request.reason == ResetReason::manual) {
                                ++reset_statistics.manual_resets;
                            }
                            if (request.reason ==
                                ResetReason::invalid_game_state) {
                                ++reset_statistics.invalid_state_resets;
                            }
                            pending_training_reset_request = nlohmann::json{
                                {"protocol_version", kServerProtocolVersion},
                                {"request_id", request.metadata.request_id},
                                {"timestamp_ms", epoch_milliseconds()},
                                {"type", "reset_episode"},
                                {"reset_id", reset_id},
                                {"episode_id", request.episode_id},
                                {"reason", reset_reason_slug(request.reason)},
                                {"state", "requested"},
                                {"load_save_state", true},
                                {"discard_inflight_action", true},
                                {"settle_delay_ms", 0},
                            };
                            pending_training_reset_attempt = 0;
                            next_training_reset_retry =
                                Clock::time_point::min();
                            send_telemetry_marker(
                                "reset_requested",
                                request.episode_id,
                                static_cast<float>(reset_id),
                                std::string(reset_reason_slug(request.reason))
                            );
                        }
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", request.metadata.request_id},
                            {"timestamp_ms", epoch_milliseconds()},
                            {"type", result.accepted
                                ? "request_reset_accepted"
                                : "error"},
                            {"reset_id", result.reset_id},
                            {"episode_id", request.episode_id},
                            {"aborted", result.episode_aborted},
                            {"message", result.message},
                        });
                        continue;
                    }
                    if (std::holds_alternative<ResetAck>(control_message)) {
                        const auto& acknowledgement =
                            std::get<ResetAck>(control_message);
                        ResetTransition result;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            result = protocol_session.acknowledge_reset(
                                acknowledgement
                            );
                        }
                        if (result.accepted && !result.duplicate) {
                            send_telemetry_marker(
                                "reset_executed",
                                acknowledgement.episode_id,
                                static_cast<float>(acknowledgement.reset_id),
                                "client command executed"
                            );
                        }
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id",
                             acknowledgement.metadata.request_id},
                            {"timestamp_ms", epoch_milliseconds()},
                            {"type", result.accepted
                                ? "reset_acknowledged"
                                : "error"},
                            {"reset_id", acknowledgement.reset_id},
                            {"state", reset_state_slug(result.state)},
                            {"duplicate", result.duplicate},
                            {"message", result.message},
                        });
                        continue;
                    }
                    if (std::holds_alternative<ResetReady>(control_message)) {
                        const auto& ready =
                            std::get<ResetReady>(control_message);
                        ResetTransition result;
                        AgentMode effective_mode;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            result = protocol_session.mark_reset_ready(ready);
                            effective_mode = protocol_session.mode();
                        }
                        if (result.accepted && !result.duplicate) {
                            ++reset_statistics.resets_completed;
                            consecutive_reset_failures = 0;
                            if (pending_training_reset_request.has_value()) {
                                const float latency = static_cast<float>(
                                    epoch_milliseconds() -
                                    pending_training_reset_request
                                        ->value<std::int64_t>(
                                            "timestamp_ms",
                                            epoch_milliseconds()
                                        )
                                );
                                reset_statistics.last_reset_latency_ms =
                                    latency;
                                const float count = static_cast<float>(
                                    reset_statistics.resets_completed
                                );
                                reset_statistics.mean_reset_latency_ms +=
                                    (latency -
                                     reset_statistics.mean_reset_latency_ms) /
                                    count;
                            }
                            reinforcement_agent->reset_episode(false);
                            reinforcement_agent->set_agent_mode(effective_mode);
                            temporal_agent->reset();
                            fight_tracker->reset();
                            encounter_configuration.current_stage = 0;
                            fight_tracker->set_external_encounter_lifecycle(
                                requires_external_encounter_lifecycle(
                                    encounter_configuration
                                )
                            );
                            if (training_reset_tracker != nullptr) {
                                training_reset_tracker->reset();
                            }
                            pending_training_reset_request.reset();
                            pending_training_reset_detected_reason.reset();
                            pending_training_reset_attempt = 0;
                            active_opponent_character.store(
                                encounter_configuration.opponents[0]
                            );
                            reinforcement_agent->configure_encounter_stage(
                                encounter_configuration.opponents[0]
                            );
                            send_telemetry_marker(
                                "reset_ready",
                                result.new_episode_id,
                                static_cast<float>(ready.reset_id),
                                "stable initial state"
                            );
                        }
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", ready.metadata.request_id},
                            {"timestamp_ms", epoch_milliseconds()},
                            {"type", result.accepted
                                ? "episode_started"
                                : "error"},
                            {"episode_id", result.new_episode_id},
                            {"previous_episode_id", result.episode_id},
                            {"reset_id", ready.reset_id},
                            {"effective_mode", agent_mode_slug(effective_mode)},
                            {"effective_epsilon",
                             effective_mode == AgentMode::evaluation
                                ? 0.0f
                                : reinforcement_agent->stats().epsilon},
                            {"message", result.message},
                        });
                        continue;
                    }
                    if (std::holds_alternative<ResetFailed>(control_message)) {
                        const auto& failure =
                            std::get<ResetFailed>(control_message);
                        ResetTransition result;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            result = protocol_session.mark_reset_failed(
                                failure
                            );
                        }
                        if (result.accepted && !result.duplicate) {
                            ++reset_statistics.resets_failed;
                            ++consecutive_reset_failures;
                            pending_training_reset_request.reset();
                            pending_training_reset_detected_reason.reset();
                            pending_training_reset_attempt = 0;
                            next_training_reset_retry =
                                Clock::time_point::min();
                            metrics.write({
                                {"protocol_version",
                                 kServerProtocolVersion},
                                {"event", "reset_failed"},
                                {"timestamp_ms", epoch_milliseconds()},
                                {"connection_id", connection_id},
                                {"reset_id", failure.reset_id},
                                {"episode_id", result.episode_id},
                                {"reason", failure.reason},
                                {"recoverable", true},
                                {"retry_after_ms",
                                 options
                                     .training_reset_non_fight_timeout
                                     .count()},
                            });
                            std::cerr
                                << "[TRAINING_RESET] reset_id="
                                << failure.reset_id
                                << " connection=" << connection_id
                                << " failed: " << failure.reason
                                << "; recovery enabled\n";
                            if (consecutive_reset_failures >=
                                kResetFailuresBeforeReconnect) {
                                reset_reconnect_required = true;
                                reconnect_reset_id = failure.reset_id;
                                reconnect_episode_id = result.episode_id;
                                reconnect_failure_reason = failure.reason;
                            }
                        }
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", failure.metadata.request_id},
                            {"timestamp_ms", epoch_milliseconds()},
                            {"type", result.accepted
                                ? "reset_failed_ack"
                                : "error"},
                            {"reset_id", failure.reset_id},
                            {"state", reset_state_slug(result.state)},
                            {"recoverable", result.accepted},
                            {"retry_after_ms",
                             options.training_reset_non_fight_timeout.count()},
                            {"message", result.message},
                        });
                    }
                }
                if (reset_reconnect_required) {
                    require_client_reconnect(
                        reconnect_reset_id,
                        reconnect_episode_id,
                        reconnect_failure_reason
                    );
                    break;
                }
                if (v21_session) {
                    ResetTransition timeout;
                    {
                        std::lock_guard lock(protocol_session_mutex);
                        timeout = protocol_session.check_reset_timeout(
                            decision_tick_ms,
                            options.reset_ack_timeout.count(),
                            options.reset_ready_timeout.count()
                        );
                    }
                    if (timeout.accepted &&
                        timeout.state == ResetState::failed) {
                        ++reset_statistics.resets_failed;
                        ++reset_statistics.reset_timeouts;
                        ++consecutive_reset_failures;
                        pending_training_reset_request.reset();
                        pending_training_reset_detected_reason.reset();
                        pending_training_reset_attempt = 0;
                        next_training_reset_retry =
                            Clock::time_point::min();
                        writer.send_json({
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", telemetry_request_id++},
                            {"timestamp_ms", decision_tick_ms},
                            {"type", "reset_failed"},
                            {"reset_id", timeout.reset_id},
                            {"episode_id", timeout.episode_id},
                            {"reason", timeout.message},
                            {"recoverable", true},
                            {"retry_after_ms",
                             options.training_reset_non_fight_timeout.count()},
                        });
                        metrics.write({
                            {"protocol_version", kServerProtocolVersion},
                            {"event", "reset_timeout"},
                            {"timestamp_ms", decision_tick_ms},
                            {"connection_id", connection_id},
                            {"reset_id", timeout.reset_id},
                            {"episode_id", timeout.episode_id},
                            {"reason", timeout.message},
                            {"recoverable", true},
                            {"retry_after_ms",
                             options.training_reset_non_fight_timeout.count()},
                        });
                        if (consecutive_reset_failures >=
                            kResetFailuresBeforeReconnect) {
                            require_client_reconnect(
                                timeout.reset_id,
                                timeout.episode_id,
                                timeout.message
                            );
                            break;
                        }
                    }
                }
                FightControlSnapshot fight;
                CombatStateSnapshot combat;
                std::optional<TrainingResetTrigger>
                    training_reset_trigger;
                if (fight_tracker != nullptr) {
                    fight = fight_tracker->consume(decision_tick_ms);
                    if (v21_session) {
                        std::lock_guard lock(protocol_session_mutex);
                        if (fight.episode_started &&
                            protocol_session.episode_id() == 0) {
                            protocol_session.begin_episode(
                                std::max<std::uint64_t>(
                                    1U, fight.episode_id
                                ),
                                decision_tick_ms
                            );
                        }
                        if (protocol_session.episode_id() != 0) {
                            fight.episode_id =
                                protocol_session.episode_id();
                        }
                    }
                    if (combat_tracker != nullptr) {
                        combat =
                            combat_tracker->consume(decision_tick_ms);
                    }
                    if (training_reset_tracker != nullptr) {
                        training_reset_trigger =
                            training_reset_tracker->observe(
                                fight,
                                decision_tick_ms
                            );
                    }
                    if (fight.episode_ended && temporal_agent != nullptr) {
                        temporal_agent->reset();
                    }
                }
                const bool repeated_frame_paused =
                    repeated_frame_training_paused.load();
                if (
                    repeated_frame_paused !=
                        inference_repeated_frame_paused
                ) {
                    inference_repeated_frame_paused =
                        repeated_frame_paused;
                    if (
                        repeated_frame_paused &&
                        reinforcement_agent != nullptr
                    ) {
                        reinforcement_agent->suspend();
                    }
                    if (temporal_agent != nullptr) {
                        temporal_agent->reset();
                    }
                }
                const std::uint64_t repeated_reset_generation =
                    repeated_frame_reset_generation.load();
                if (
                    repeated_reset_generation >
                        handled_repeated_frame_reset_generation
                ) {
                    handled_repeated_frame_reset_generation =
                        repeated_reset_generation;
                    if (
                        training_reset_tracker != nullptr &&
                        !training_reset_trigger.has_value()
                    ) {
                        training_reset_trigger = TrainingResetTrigger{
                            .episode_id = fight.episode_id,
                            .frame_id =
                                repeated_frame_reset_frame_id.load(),
                            .reason = "repeated_frame_timeout",
                        };
                    }
                }
                if (
                    !v21_session &&
                    pending_training_reset_request.has_value() &&
                    last_training_reset_ack_id.load() >=
                        pending_training_reset_request
                            ->at("reset_id")
                            .get<std::uint64_t>()
                ) {
                    pending_training_reset_request.reset();
                    pending_training_reset_detected_reason.reset();
                    pending_training_reset_attempt = 0;
                }
                const bool protocol_reset_pending = [&] {
                    if (!v21_session) {
                        return false;
                    }
                    std::lock_guard lock(protocol_session_mutex);
                    return protocol_session.decisions_blocked();
                }();
                const bool training_reset_pending = v21_session
                    ? protocol_reset_pending
                    : pending_training_reset_request.has_value() ||
                        last_training_reset_id.load() >
                            last_training_reset_ack_id.load();
                const bool player_actionable_now =
                    fight.action_allowed && combat.spatial_valid &&
                    combat.player_on_ground && !combat.player_attacking &&
                    !repeated_frame_paused && !training_reset_pending;
                if (player_actionable_now) {
                    player_actionable_ticks = std::min(
                        player_actionable_ticks + 1U,
                        options.sequence_actionable_ticks
                    );
                } else {
                    player_actionable_ticks = 0;
                }
                const bool player_action_stable =
                    player_actionable_ticks >=
                        options.sequence_actionable_ticks;
                std::optional<TemporalAgentOutput> temporal_output;
                const bool temporal_control_allowed =
                    (
                        reinforcement_agent == nullptr ||
                        fight.action_allowed
                    ) && !training_reset_pending &&
                    !repeated_frame_paused;
                if (temporal_agent != nullptr && temporal_control_allowed &&
                    temporal_agent->ready()) {
                    try {
                        temporal_output =
                            temporal_agent->advance_and_infer(
                                latest_applied,
                                current_decision_id,
                                decision_tick_ms
                            );
                    } catch (const std::exception& error) {
                        temporal_model_errors.fetch_add(1);
                        std::cerr << "[TEMPORAL] connection=" << connection_id
                                  << " inference failed: " << error.what()
                                  << '\n';
                    }
                }
                MockAgentDecision mock_decision;
                ReinforcementDecision reinforcement_decision;
                std::vector<float> reinforcement_state;
                std::vector<int> commands;
                bool reinforcement_step_ok = true;
                if (reinforcement_agent != nullptr) {
                    if (temporal_output.has_value()) {
                        reinforcement_state = make_reinforcement_state(
                            *temporal_output,
                            fight,
                            combat,
                            latest_applied,
                            encounter_configuration
                        );
                    }
                    const float total_reward =
                        fight.reward + combat.shaping_reward;
                    try {
                        // The file is intentionally read on the decision path
                        // so edits become effective without a restart. The
                        // agent applies the new filter only at an option
                        // boundary, never in the middle of a timed program.
                        reinforcement_agent->set_directed_action_curriculum(
                            load_directed_curriculum(
                                options.directed_curriculum_path,
                                player_character
                            )
                        );
                        reinforcement_decision = reinforcement_agent->step({
                            .state = reinforcement_state,
                            .temporal_visual_feature =
                                temporal_output.has_value()
                                    ? std::span<const float>(
                                          temporal_output
                                              ->visual_feature
                                      )
                                    : std::span<const float>{},
                            .temporal_jpeg =
                                temporal_output.has_value()
                                    ? std::span<const std::uint8_t>(
                                          temporal_output->visual_jpeg
                                      )
                                    : std::span<const std::uint8_t>{},
                            .temporal_controller_features =
                                temporal_output.has_value()
                                    ? std::span<const float>(
                                          temporal_output
                                              ->controller_features
                                      )
                                    : std::span<const float>{},
                            .reward = total_reward,
                            .action_allowed =
                                fight.action_allowed &&
                                temporal_output.has_value() &&
                                !training_reset_pending &&
                                !repeated_frame_paused,
                            .episode_started = fight.episode_started,
                            .episode_ended = fight.episode_ended,
                            .terminal = fight.terminal,
                            .episode_won =
                                fight.episode_ended &&
                                (fight.transition_reason == "round_won" ||
                                 fight.transition_reason == "encounter_won"),
                            .episode_id = fight.episode_id,
                            .frame_id = fight.frame_id,
                            .decision_id = current_decision_id,
                            .fighter_distance = combat.distance,
                            .player_x = combat.player_x,
                            .opponent_x = combat.opponent_x,
                            .spatial_valid = combat.spatial_valid,
                            .player_on_ground = combat.player_on_ground,
                            .opponent_on_ground = combat.opponent_on_ground,
                            .player_attacking = combat.player_attacking,
                            .opponent_attacking =
                                combat.opponent_attacking,
                            .player_action_stable = player_action_stable,
                        });
                        commands = reinforcement_decision.commands;
                    } catch (const std::exception& error) {
                        reinforcement_step_ok = false;
                        reinforcement_errors.fetch_add(1);
                        commands.clear();
                        std::cerr
                            << "[RL] connection=" << connection_id
                            << " decision=" << current_decision_id
                            << " failed; controls released: "
                            << error.what() << '\n';
                    }
                    if (repeated_frame_paused) {
                        commands.clear();
                    }
                } else {
                    mock_decision = agent.infer(observation, applied_actions);
                    commands = mock_decision.commands;
                }
                if (reinforcement_agent != nullptr) {
                    AgentMode effective_mode =
                        options.reinforcement_training_enabled
                            ? AgentMode::training
                            : AgentMode::evaluation;
                    if (v21_session) {
                        std::lock_guard lock(protocol_session_mutex);
                        effective_mode = protocol_session.mode();
                    }
                    if (fight.episode_started ||
                        episode_statistics.episode_id != fight.episode_id) {
                        episode_statistics = EpisodeStatistics{
                            .episode_id = fight.episode_id,
                            .mode = effective_mode,
                            .best_step_reward =
                                -std::numeric_limits<float>::infinity(),
                            .worst_step_reward =
                                std::numeric_limits<float>::infinity(),
                            .started_timestamp_ms = decision_tick_ms,
                        };
                    }
                    if (!training_reset_pending &&
                        episode_statistics.episode_id != 0) {
                        const float step_reward = reinforcement_step_ok
                            ? reinforcement_decision.reward_consumed
                            : fight.reward + combat.shaping_reward;
                        episode_statistics.reward_sum += step_reward;
                        episode_statistics.episode_return =
                            reinforcement_decision.episode_return;
                        episode_statistics.best_step_reward = std::max(
                            episode_statistics.best_step_reward, step_reward
                        );
                        episode_statistics.worst_step_reward = std::min(
                            episode_statistics.worst_step_reward, step_reward
                        );
                        episode_statistics.damage_dealt += fight.damage_dealt;
                        episode_statistics.damage_received += fight.damage_taken;
                        ++episode_statistics.decisions;
                        const bool exploratory =
                            reinforcement_decision.selection_source.starts_with(
                                "epsilon_"
                            );
                        episode_statistics.exploratory_decisions +=
                            exploratory ? 1U : 0U;
                        episode_statistics.greedy_decisions +=
                            exploratory ? 0U : 1U;
                        episode_statistics.epsilon_sum +=
                            reinforcement_decision.epsilon;
                        if (stage_completed_this_tick) {
                            ++episode_statistics.stages_completed;
                        }
                    }
                    reward_log_accumulator += reinforcement_step_ok
                        ? reinforcement_decision.reward_consumed
                        : fight.reward + combat.shaping_reward;
                    terminal_reward_log_accumulator +=
                        fight.terminal_reward;
                    damage_dealt_log_accumulator += fight.damage_dealt;
                    damage_taken_log_accumulator += fight.damage_taken;
                    shaping_reward_log_accumulator +=
                        combat.shaping_reward;
                    combo_reward_log_accumulator +=
                        combat.combo_reward;
                    defense_reward_log_accumulator +=
                        combat.defense_reward;
                    idle_penalty_log_accumulator +=
                        combat.idle_penalty;
                    corner_penalty_log_accumulator +=
                        combat.corner_penalty;
                    neutral_penalty_log_accumulator +=
                        reinforcement_decision.neutral_penalty;
                    ++reward_log_decisions;
                    reward_log_active_decisions +=
                        fight.action_allowed && !repeated_frame_paused
                            ? 1U
                            : 0U;
                    const auto reward_log_now = Clock::now();
                    if (reward_log_now >= next_reward_log ||
                        fight.episode_ended) {
                        const ReinforcementAgentStats stats =
                            reinforcement_agent->stats();
                        std::cout
                            << "\n\n[RL_REWARD] reward_gained="
                            << reward_log_accumulator
                            << " reward_accumulated="
                            << stats.current_episode_return
                            << " reward_terminal="
                            << terminal_reward_log_accumulator
                            << " result="
                            << (
                                fight.episode_ended
                                    ? fight.transition_reason
                                    : "ongoing"
                            )
                            << " character="
                            << fighter_character_slug(player_character)
                            << " opponent="
                            << opponent_character_slug(
                                   active_opponent_character.load()
                               )
                            << " connection=" << connection_id
                            << " episode=" << fight.episode_id
                            << " phase=" << fight_phase_name(fight.phase)
                            << " interval_ms="
                            << options.reinforcement_reward_log_interval.count()
                            << " damage_dealt="
                            << damage_dealt_log_accumulator
                            << " damage_taken="
                            << damage_taken_log_accumulator
                            << " shaping="
                            << shaping_reward_log_accumulator
                            << " combo_reward="
                            << combo_reward_log_accumulator
                            << " defense_reward="
                            << defense_reward_log_accumulator
                            << " idle_penalty="
                            << idle_penalty_log_accumulator
                            << " corner_penalty="
                            << corner_penalty_log_accumulator
                            << " neutral_penalty="
                            << neutral_penalty_log_accumulator
                            << " hp=" << fight.player_health << '/'
                            << fight.opponent_health
                            << " x=" << combat.player_x << '/'
                            << combat.opponent_x
                            << " distance=" << combat.distance
                            << " combo=" << combat.consecutive_hits
                            << " idle_ms="
                            << combat.stationary_time_ms
                            << " corner_ms=" << combat.corner_time_ms
                            << " active_ticks="
                            << reward_log_active_decisions << '/'
                            << reward_log_decisions
                            << " epsilon=" << stats.epsilon
                            << " replay=" << stats.replay_size
                            << " winning_replay="
                            << stats.winning_replay_size
                            << " winning_episodes="
                            << stats.winning_episodes
                            << " winning_target="
                            << stats.effective_winning_replay_fraction
                            << " matchup_selections="
                            << stats.matchup_action_selections
                            << " matchup_episodes="
                            << stats.matchup_episodes_completed
                            << " matchup_wins="
                            << stats.matchup_wins
                            << " matchups="
                            << stats.shared_replay_matchups
                            << " updates=" << stats.training_updates
                            << " loss=" << stats.last_training_loss
                            << " q_pred="
                            << stats.last_predicted_q_mean
                            << " td_target="
                            << stats.last_td_target_mean
                            << " q_gap="
                            << stats.last_q_target_gap
                            << " abs_td="
                            << stats.last_abs_td_error
                            << " bootstrap="
                            << stats.last_bootstrap_mean
                            << " target_clipped="
                            << stats.last_q_target_clipped_fraction
                            << " terminal_batch="
                            << stats.last_terminal_sample_fraction
                            << " terminal_q_pred="
                            << stats.last_terminal_predicted_q_mean
                            << " terminal_target="
                            << stats.last_terminal_td_target_mean
                            << " critic_disagreement="
                            << stats.last_critic_disagreement
                            << " hard_negative_batch="
                            << stats.last_hard_negative_sample_fraction
                            << " winning_batch="
                            << stats.last_winning_sample_fraction
                            << " greedy_episode="
                            << reinforcement_decision
                                   .greedy_training_episode
                            << " option=\""
                            << displayed_action_name(
                                   reinforcement_decision.option_name,
                                   reinforcement_decision.commands_mirrored
                               ) << '"'
                            << "\n\n" << std::flush;
                        reward_log_accumulator = 0.0f;
                        terminal_reward_log_accumulator = 0.0f;
                        damage_dealt_log_accumulator = 0.0f;
                        damage_taken_log_accumulator = 0.0f;
                        shaping_reward_log_accumulator = 0.0f;
                        combo_reward_log_accumulator = 0.0f;
                        defense_reward_log_accumulator = 0.0f;
                        idle_penalty_log_accumulator = 0.0f;
                        corner_penalty_log_accumulator = 0.0f;
                        neutral_penalty_log_accumulator = 0.0f;
                        reward_log_decisions = 0;
                        reward_log_active_decisions = 0;
                        next_reward_log = reward_log_now +
                            options.reinforcement_reward_log_interval;
                    }
                }
                const auto inference_end = Clock::now();
                const std::int64_t now_ms = epoch_milliseconds();
                const double inference_ms = std::chrono::duration<double, std::milli>(
                    inference_end - inference_start
                ).count();
                const auto& latest = observation.back();
                const std::int64_t latency_ms = latest->client_timestamp_ms > 0
                    ? now_ms - latest->client_timestamp_ms
                    : -1;
                const std::int64_t server_frame_age_ms =
                    now_ms - latest->received_timestamp_ms;
                std::optional<nlohmann::json>
                    frame_stream_control_message;
                const bool frame_stream_stalled =
                    server_frame_age_ms >=
                        options.frame_stream_stall_timeout.count();
                if (frame_stream_stalled) {
                    if (!frame_stream_recovery_active) {
                        frame_stream_recovery_active = true;
                        ++frame_stream_recovery_id;
                        frame_stream_recovery_last_frame_id =
                            latest->frame_id;
                        frame_stream_recovery_attempt = 0;
                        frame_stream_recovery_started_ms = now_ms;
                        next_frame_stream_recovery =
                            Clock::time_point::min();
                        if (temporal_agent != nullptr) {
                            temporal_agent->reset();
                        }
                        if (reinforcement_agent != nullptr) {
                            reinforcement_agent->suspend();
                        }
                        metrics.write({
                            {"protocol_version", kServerProtocolVersion},
                            {"event", "frame_stream_stalled"},
                            {"timestamp_ms", now_ms},
                            {"connection_id", connection_id},
                            {"recovery_id", frame_stream_recovery_id},
                            {"last_frame_id", latest->frame_id},
                            {"frame_age_ms", server_frame_age_ms},
                        });
                        std::cerr
                            << "[FRAME_STREAM] connection=" << connection_id
                            << " recovery_id=" << frame_stream_recovery_id
                            << " stalled at frame=" << latest->frame_id
                            << " age_ms=" << server_frame_age_ms
                            << "; requesting in-place capture restart\n";
                    }
                    const auto recovery_now = Clock::now();
                    if (recovery_now >= next_frame_stream_recovery) {
                        ++frame_stream_recovery_attempt;
                        frame_stream_control_message = nlohmann::json{
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", telemetry_request_id++},
                            {"timestamp_ms", now_ms},
                            {"type", "frame_stream_recovery"},
                            {"recovery_id", frame_stream_recovery_id},
                            {"reason", "no_new_frames"},
                            {"last_frame_id",
                             frame_stream_recovery_last_frame_id},
                            {"frame_age_ms", server_frame_age_ms},
                            {"attempt", frame_stream_recovery_attempt},
                            {"restart_capture_pipeline", true},
                            {"clear_pending_frames", true},
                            {"preserve_websocket", true},
                            {"release_controls", true},
                            {"cancel_pending_sequence", true},
                            {"ack_required", false},
                        };
                        next_frame_stream_recovery = recovery_now +
                            options.frame_stream_recovery_interval;
                    }
                } else if (frame_stream_recovery_active) {
                    const std::int64_t stalled_duration_ms =
                        now_ms - frame_stream_recovery_started_ms;
                    frame_stream_recovery_active = false;
                    frame_stream_recovery_attempt = 0;
                    frame_stream_control_message = nlohmann::json{
                        {"protocol_version", kServerProtocolVersion},
                        {"request_id", telemetry_request_id++},
                        {"timestamp_ms", now_ms},
                        {"type", "frame_stream_resumed"},
                        {"recovery_id", frame_stream_recovery_id},
                        {"frame_id", latest->frame_id},
                        {"stalled_duration_ms", stalled_duration_ms},
                        {"websocket_preserved", true},
                    };
                    metrics.write({
                        {"protocol_version", kServerProtocolVersion},
                        {"event", "frame_stream_resumed"},
                        {"timestamp_ms", now_ms},
                        {"connection_id", connection_id},
                        {"recovery_id", frame_stream_recovery_id},
                        {"frame_id", latest->frame_id},
                        {"stalled_duration_ms", stalled_duration_ms},
                    });
                    std::cout
                        << "[FRAME_STREAM] connection=" << connection_id
                        << " recovery_id=" << frame_stream_recovery_id
                        << " resumed at frame=" << latest->frame_id
                        << " after_ms=" << stalled_duration_ms << '\n';
                }
                std::vector<std::uint64_t> frame_ids;
                frame_ids.reserve(observation.size());
                for (const auto& frame : observation) {
                    frame_ids.push_back(frame->frame_id);
                }

                std::optional<nlohmann::json> episode_summary;
                std::optional<nlohmann::json> evaluation_summary;
                if (reinforcement_agent != nullptr &&
                    fight.episode_ended && episode_statistics.episode_id != 0) {
                    episode_statistics.won =
                        fight.transition_reason == "round_won" ||
                        fight.transition_reason == "encounter_won";
                    episode_statistics.lost =
                        fight.transition_reason == "round_lost" ||
                        fight.transition_reason == "encounter_lost";
                    ModeStatistics& aggregate = mode_statistics[
                        statistics_key(episode_statistics.mode)
                    ];
                    ++aggregate.episodes;
                    aggregate.wins += episode_statistics.won ? 1U : 0U;
                    aggregate.losses += episode_statistics.lost ? 1U : 0U;
                    aggregate.aborted += episode_statistics.aborted ? 1U : 0U;
                    aggregate.stage_completions +=
                        episode_statistics.stages_completed;
                    aggregate.episode_return_sum +=
                        episode_statistics.episode_return;
                    aggregate.damage_dealt_sum +=
                        episode_statistics.damage_dealt;
                    aggregate.damage_received_sum +=
                        episode_statistics.damage_received;
                    if (aggregate.episodes == 1U) {
                        aggregate.best_episode_return =
                            episode_statistics.episode_return;
                        aggregate.worst_episode_return =
                            episode_statistics.episode_return;
                    } else {
                        aggregate.best_episode_return = std::max(
                            aggregate.best_episode_return,
                            episode_statistics.episode_return
                        );
                        aggregate.worst_episode_return = std::min(
                            aggregate.worst_episode_return,
                            episode_statistics.episode_return
                        );
                    }
                    std::vector<std::string> opponent_slugs;
                    for (std::size_t index = 0;
                         index < encounter_configuration.opponent_count;
                         ++index) {
                        opponent_slugs.emplace_back(opponent_character_slug(
                            encounter_configuration.opponents[index]
                        ));
                    }
                    const std::uint64_t duration_ms = static_cast<std::uint64_t>(
                        std::max<std::int64_t>(
                            0,
                            now_ms -
                                episode_statistics.started_timestamp_ms
                        )
                    );
                    aggregate.episode_duration_ms_sum +=
                        static_cast<double>(duration_ms);
                    episode_summary = nlohmann::json{
                        {"protocol_version", kServerProtocolVersion},
                        {"request_id", telemetry_request_id++},
                        {"timestamp_ms", now_ms},
                        {"type", "episode_summary"},
                        {"mode", agent_mode_slug(episode_statistics.mode)},
                        {"episode_id", episode_statistics.episode_id},
                        {"encounter_type",
                         encounter_type_slug(encounter_configuration.type)},
                        {"controlled_character",
                         fighter_character_slug(player_character)},
                        {"opponents", opponent_slugs},
                        {"won", episode_statistics.won},
                        {"lost", episode_statistics.lost},
                        {"aborted", episode_statistics.aborted},
                        {"stages_completed",
                         episode_statistics.stages_completed},
                        {"opponent_count",
                         encounter_configuration.opponent_count},
                        {"episode_return",
                         episode_statistics.episode_return},
                        {"total_damage_dealt",
                         episode_statistics.damage_dealt},
                        {"total_damage_received",
                         episode_statistics.damage_received},
                        {"action_selections",
                         episode_statistics.decisions},
                        {"exploratory_selections",
                         episode_statistics.exploratory_decisions},
                        {"mean_epsilon",
                         episode_statistics.decisions == 0
                            ? 0.0
                            : episode_statistics.epsilon_sum /
                                static_cast<double>(
                                    episode_statistics.decisions
                                )},
                        {"duration_ms", duration_ms},
                    };
                    if (episode_statistics.mode == AgentMode::evaluation) {
                        evaluation_summary = nlohmann::json{
                            {"protocol_version", kServerProtocolVersion},
                            {"request_id", telemetry_request_id++},
                            {"timestamp_ms", now_ms},
                            {"type", "evaluation_summary"},
                            {"session_id", evaluation_session_id},
                            {"episodes", aggregate.episodes},
                            {"wins", aggregate.wins},
                            {"losses", aggregate.losses},
                            {"win_rate", aggregate.episodes == 0
                                ? 0.0
                                : static_cast<double>(aggregate.wins) /
                                    static_cast<double>(aggregate.episodes)},
                            {"mean_return", aggregate.episodes == 0
                                ? 0.0
                                : aggregate.episode_return_sum /
                                    static_cast<double>(aggregate.episodes)},
                            {"best_return",
                             aggregate.best_episode_return},
                            {"mean_damage_dealt", aggregate.episodes == 0
                                ? 0.0
                                : aggregate.damage_dealt_sum /
                                    static_cast<double>(aggregate.episodes)},
                            {"mean_damage_received", aggregate.episodes == 0
                                ? 0.0
                                : aggregate.damage_received_sum /
                                    static_cast<double>(aggregate.episodes)},
                            {"mean_episode_duration_ms",
                             aggregate.episodes == 0
                                ? 0.0
                                : aggregate.episode_duration_ms_sum /
                                    static_cast<double>(aggregate.episodes)},
                        };
                    }
                }

                const bool timed_sequence_active =
                    options.timed_sequences_enabled &&
                    (
                        reinforcement_agent != nullptr
                            ? reinforcement_decision.scripted_sequence
                            : mock_decision.step_count > 0U
                    );
                const bool timed_sequence_started =
                    timed_sequence_active &&
                    (
                        reinforcement_agent != nullptr
                            ? reinforcement_decision.sequence_started
                            : mock_decision.sequence_started
                    );
                const std::uint64_t timed_sequence_id =
                    reinforcement_agent != nullptr
                        ? reinforcement_decision.sequence_id
                        : current_decision_id -
                            static_cast<std::uint64_t>(
                                mock_decision.step_index - 1U
                            );
                const std::string timed_sequence_name =
                    reinforcement_agent != nullptr
                        ? displayed_action_name(
                              reinforcement_decision.option_name,
                              reinforcement_decision.commands_mirrored
                          )
                        : mock_decision.move_name;
                const auto& timed_sequence_commands =
                    reinforcement_agent != nullptr
                        ? reinforcement_decision.sequence_commands
                        : mock_decision.sequence_commands;

                nlohmann::json response{
                    {"protocol_version", kActionProtocolVersion},
                    {"request_id", telemetry_request_id++},
                    {"type", "action"},
                    {"action_mode", kActionMode},
                    {"supported_action_modes", {
                        "state", "timed_sequence"
                    }},
                    {"timed_sequences_enabled",
                     options.timed_sequences_enabled},
                    {"timed_sequence_version", 2},
                    {"timed_sequence_step_semantics",
                     "exact_controller_state"},
                    {"sequence_input_hold_ms",
                     options.sequence_input_hold.count()},
                    {"sequence_neutral_hold_ms",
                     options.sequence_neutral_hold.count()},
                    {"sequence_final_release_ms",
                     options.sequence_final_release.count()},
                    {"sequence_actionable_ticks",
                     options.sequence_actionable_ticks},
                    {"player_character",
                     fighter_character_slug(player_character)},
                    {"opponent_character",
                     opponent_character_slug(
                         active_opponent_character.load()
                     )},
                    {"opponent_character_name",
                     opponent_character_name(
                         active_opponent_character.load()
                     )},
                    {"encounter_type",
                     encounter_type_slug(encounter_configuration.type)},
                    {"encounter_event",
                     encounter_event_slug(latest_encounter_event)},
                    {"encounter_stage",
                     encounter_configuration.current_stage},
                    {"opponent_count",
                     encounter_configuration.opponent_count},
                    {"remaining_opponents",
                     encounter_configuration.opponent_count -
                        encounter_configuration.current_stage - 1U},
                    {"commands", commands},
                    {"mock_move", mock_decision.move_name},
                    {"mock_step", mock_decision.step_index},
                    {"mock_steps", mock_decision.step_count},
                    {"curriculum_source", mock_decision.action_source},
                    {"curriculum_phase", mock_decision.phase},
                    {"mock_step_duration_ms", options.decision_interval.count()},
                    {"decision_id", current_decision_id},
                    {"frame_id", latest->frame_id},
                    {"frame_ids", frame_ids},
                    {"timestamp_ms", now_ms},
                    {"decision_tick_timestamp_ms", decision_tick_ms},
                    {"control_interval_ms", options.decision_interval.count()},
                    {"failsafe_timeout_ms", options.failsafe_timeout.count()},
                    {"metrics", {
                        {"frame_latency_ms", latency_ms},
                        {"server_frame_age_ms", server_frame_age_ms},
                        {"frame_stream_stalled", frame_stream_stalled},
                        {"frame_stream_recovery_active",
                         frame_stream_recovery_active},
                        {"frame_stream_recovery_id",
                         frame_stream_recovery_active
                            ? frame_stream_recovery_id : 0U},
                        {"inference_ms", inference_ms},
                        {"buffer_size", buffer.size()},
                        {"frames_received", received_frames.load()},
                        {"frames_dropped", buffer.dropped_total()},
                        {"decode_errors", decode_errors.load()},
                        {"client_message_errors", client_message_errors.load()},
                        {"acknowledgements_received", acknowledgements_received.load()},
                        {"acknowledgements_rejected", acknowledgements_rejected.load()},
                        {"applied_action_history_size", applied_actions.size()},
                        {"last_applied_decision_id",
                         action_history.last_applied_decision_id()},
                        {"temporal_model_enabled", temporal_agent != nullptr},
                        {"temporal_model_ready", temporal_output.has_value()},
                        {"temporal_model_errors", temporal_model_errors.load()},
                        {"temporal_frames_processed",
                         temporal_frames_processed.load()},
                        {"temporal_frames_overwritten",
                         temporal_frames_overwritten.load()},
                        {"persistence_frames_saved",
                         persistence_frames_saved.load()},
                        {"persistence_frames_dropped",
                         persistence_frames_dropped.load()},
                        {"persistence_errors", persistence_errors.load()},
                    }},
                };
                if (timed_sequence_active) {
                    response["action_mode"] = "timed_sequence";
                    response["sequence_id"] = timed_sequence_id;
                    response["sequence_status"] =
                        timed_sequence_started ? "start" : "running";
                    response["commands_role"] = "expected_state_only";
                    if (timed_sequence_started) {
                        response["timed_sequence"] =
                            make_timed_sequence_payload(
                                timed_sequence_id,
                                timed_sequence_name,
                                timed_sequence_commands,
                                options
                            );
                    }
                } else {
                    response["commands_role"] = "apply_immediately";
                }
                if (reinforcement_agent != nullptr) {
                    const ReinforcementAgentStats rl_stats =
                        reinforcement_agent->stats();
                    AgentMode effective_mode =
                        options.reinforcement_training_enabled
                            ? AgentMode::training
                            : AgentMode::evaluation;
                    ExplorationMode effective_exploration =
                        ExplorationMode::automatic;
                    ResetState effective_reset_state = ResetState::idle;
                    std::uint64_t effective_reset_id = 0;
                    bool mode_change_pending = false;
                    if (v21_session) {
                        std::lock_guard lock(protocol_session_mutex);
                        effective_mode = protocol_session.mode();
                        effective_exploration =
                            protocol_session.exploration_mode();
                        effective_reset_state =
                            protocol_session.reset_state();
                        effective_reset_id =
                            protocol_session.active_reset_id();
                        mode_change_pending =
                            protocol_session.pending_mode().has_value();
                    }
                    response["agent_mode"] = v21_session
                        ? agent_mode_slug(effective_mode)
                        : "reinforcement_learning";
                    response["agent_controller"] =
                        "reinforcement_learning";
                    response["exploration_mode"] =
                        exploration_mode_slug(effective_exploration);
                    response["control_enabled"] =
                        fight.action_allowed && temporal_output.has_value() &&
                        reinforcement_step_ok && !training_reset_pending &&
                        !repeated_frame_paused;
                    response["cancel_pending_sequence"] =
                        !fight.action_allowed || !reinforcement_step_ok ||
                        training_reset_pending || repeated_frame_paused;
                    response["fight_phase"] = fight_phase_name(fight.phase);
                    response["episode_id"] = fight.episode_id;
                    response["episode_ended"] = fight.episode_ended;
                    response["terminal"] = fight.terminal;
                    response["player_health"] = fight.player_health;
                    response["opponent_health"] = fight.opponent_health;
                    response["reward"] = reinforcement_step_ok
                        ? reinforcement_decision.reward_consumed
                        : fight.reward + combat.shaping_reward;
                    response["terminal_reward"] =
                        fight.terminal_reward;
                    response["reward_breakdown"] = {
                        {"damage_dealt", fight.damage_dealt},
                        {"damage_received", fight.damage_taken},
                        {"stage_bonus", latest_stage_bonus},
                        {"victory_bonus",
                         fight.transition_reason == "encounter_won"
                            ? fight.terminal_reward : 0.0f},
                        {"defeat_penalty",
                         fight.transition_reason == "encounter_lost"
                            ? -fight.terminal_reward : 0.0f},
                        {"idle_penalty", combat.idle_penalty},
                        {"useless_action_penalty", combat.corner_penalty},
                        {"neutral_penalty",
                         reinforcement_decision.neutral_penalty},
                        {"opponent_health_delta_ignored",
                         opponent_health_delta_ignored},
                        {"step_total", reinforcement_step_ok
                            ? reinforcement_decision.reward_consumed
                            : fight.reward + combat.shaping_reward},
                        {"episode_return",
                         reinforcement_decision.episode_return},
                    };
                    response["rl_option"] = displayed_action_name(
                        reinforcement_decision.option_name,
                        reinforcement_decision.commands_mirrored
                    );
                    response["rl_option_base"] =
                        reinforcement_decision.option_name;
                    response["rl_option_index"] =
                        reinforcement_decision.option_index;
                    response["rl_option_step"] =
                        reinforcement_decision.option_step;
                    response["rl_option_steps"] =
                        reinforcement_decision.option_steps;
                    response["rl_selection_source"] =
                        reinforcement_decision.selection_source;
                    response["rl_commands_mirrored"] =
                        reinforcement_decision.commands_mirrored;
                    response["rl_greedy_training_episode"] =
                        reinforcement_decision.greedy_training_episode;
                    auto& response_metrics = response["metrics"];
                    response_metrics["hud_valid"] = fight.hud_valid;
                    response_metrics["hud_confidence"] =
                        fight.hud_confidence;
                    response_metrics["damage_dealt"] =
                        fight.damage_dealt;
                    response_metrics["damage_taken"] =
                        fight.damage_taken;
                    response_metrics["health_reward"] =
                        fight.reward - fight.terminal_reward;
                    response_metrics["combat_shaping_reward"] =
                        combat.shaping_reward;
                    response_metrics["combo_reward"] =
                        combat.combo_reward;
                    response_metrics["defense_reward"] =
                        combat.defense_reward;
                    response_metrics["idle_penalty"] =
                        combat.idle_penalty;
                    response_metrics["corner_penalty"] =
                        combat.corner_penalty;
                    response_metrics["neutral_penalty"] =
                        reinforcement_decision.neutral_penalty;
                    response_metrics["rl_allowed_options"] =
                        reinforcement_decision.allowed_options;
                    response_metrics["rl_impact_credit_step"] =
                        reinforcement_decision.impact_credit_step;
                    response_metrics["rl_impact_credit_steps"] =
                        reinforcement_decision.impact_credit_steps;
                    response_metrics["rl_disable_cloud_cooldown_remaining"] =
                        reinforcement_decision
                            .disable_cloud_cooldown_remaining;
                    response_metrics["rl_throw_cooldown_remaining"] =
                        reinforcement_decision.throw_cooldown_remaining;
                    response_metrics["rl_sweep_cooldown_remaining"] =
                        reinforcement_decision.sweep_cooldown_remaining;
                    response_metrics["combat_state_version"] = 2U;
                    response_metrics["combat_spatial_valid"] =
                        combat.spatial_valid;
                    response_metrics["combat_spatial_confidence"] =
                        combat.spatial_confidence;
                    response_metrics[
                        "combat_opponent_action_confidence"
                    ] = combat.opponent_action_confidence;
                    response_metrics["player_x"] =
                        combat.player_x;
                    response_metrics["opponent_x"] =
                        combat.opponent_x;
                    response_metrics["fighter_distance"] =
                        combat.distance;
                    response_metrics["player_on_ground"] =
                        combat.player_on_ground;
                    response_metrics["opponent_on_ground"] =
                        combat.opponent_on_ground;
                    response_metrics["player_defending"] =
                        combat.player_defending;
                    response_metrics["opponent_defending"] =
                        combat.opponent_defending;
                    response_metrics["player_attacking"] =
                        combat.player_attacking;
                    response_metrics["opponent_attacking"] =
                        combat.opponent_attacking;
                    response_metrics["time_since_last_damage_ms"] =
                        combat.time_since_last_damage_ms;
                    response_metrics["consecutive_hits"] =
                        combat.consecutive_hits;
                    response_metrics["stationary_time_ms"] =
                        combat.stationary_time_ms;
                    response_metrics["corner_time_ms"] =
                        combat.corner_time_ms;
                    response_metrics["fight_transition_reason"] =
                        fight.transition_reason;
                    response_metrics["rl_epsilon"] =
                        reinforcement_decision.epsilon;
                    response_metrics["rl_selected_q"] =
                        reinforcement_decision.selected_q;
                    response_metrics["rl_episode_return"] =
                        reinforcement_decision.episode_return;
                    response_metrics["rl_replay_size"] =
                        reinforcement_decision.replay_size;
                    response_metrics["rl_winning_replay_size"] =
                        reinforcement_decision.winning_replay_size;
                    response_metrics["rl_winning_episodes"] =
                        reinforcement_decision.winning_episodes;
                    response_metrics["rl_matchup_action_selections"] =
                        reinforcement_decision.matchup_action_selections;
                    response_metrics["rl_matchup_episodes_completed"] =
                        reinforcement_decision.matchup_episodes_completed;
                    response_metrics["rl_matchup_wins"] =
                        reinforcement_decision.matchup_wins;
                    response_metrics["rl_effective_winning_replay_fraction"] =
                        reinforcement_agent->stats()
                            .effective_winning_replay_fraction;
                    response_metrics["rl_training_updates"] =
                        reinforcement_decision.training_updates;
                    response_metrics["rl_training_loss"] =
                        reinforcement_decision.training_loss;
                    response_metrics["rl_predicted_q_mean"] =
                        reinforcement_decision.predicted_q_mean;
                    response_metrics["rl_td_target_mean"] =
                        reinforcement_decision.td_target_mean;
                    response_metrics["rl_bootstrap_mean"] =
                        reinforcement_decision.bootstrap_mean;
                    response_metrics["rl_q_target_gap"] =
                        reinforcement_decision.q_target_gap;
                    response_metrics["rl_abs_td_error"] =
                        reinforcement_decision.abs_td_error;
                    response_metrics["rl_q_target_clipped_fraction"] =
                        reinforcement_decision
                            .q_target_clipped_fraction;
                    response_metrics["rl_terminal_sample_fraction"] =
                        reinforcement_decision.terminal_sample_fraction;
                    response_metrics["rl_terminal_predicted_q_mean"] =
                        reinforcement_decision
                            .terminal_predicted_q_mean;
                    response_metrics["rl_terminal_td_target_mean"] =
                        reinforcement_decision.terminal_td_target_mean;
                    response_metrics["rl_critic_disagreement"] =
                        reinforcement_decision.critic_disagreement;
                    response_metrics[
                        "rl_hard_negative_sample_fraction"
                    ] = reinforcement_decision
                        .hard_negative_sample_fraction;
                    response_metrics[
                        "rl_winning_sample_fraction"
                    ] = reinforcement_decision
                        .winning_sample_fraction;
                    response_metrics["rl_training_performed"] =
                        reinforcement_decision.training_performed;
                    response_metrics["weights_frozen"] =
                        effective_mode == AgentMode::evaluation;
                    response_metrics["replay_frozen"] =
                        effective_mode == AgentMode::evaluation;
                    response_metrics["scheduler_frozen"] =
                        effective_mode == AgentMode::evaluation;
                    response_metrics["mode_change_pending"] =
                        mode_change_pending;
                    response_metrics["reset_state"] =
                        reset_state_slug(effective_reset_state);
                    response_metrics["active_reset_id"] =
                        effective_reset_id;
                    response_metrics["resets_requested"] =
                        reset_statistics.resets_requested;
                    response_metrics["resets_completed"] =
                        reset_statistics.resets_completed;
                    response_metrics["resets_failed"] =
                        reset_statistics.resets_failed;
                    response_metrics["reset_timeouts"] =
                        reset_statistics.reset_timeouts;
                    response_metrics["last_reset_latency_ms"] =
                        reset_statistics.last_reset_latency_ms;
                    response_metrics["mean_reset_latency_ms"] =
                        reset_statistics.mean_reset_latency_ms;
                    response_metrics["evaluation_session_id"] =
                        evaluation_session_id;
                    response_metrics["replay_capacity"] =
                        options.reinforcement_replay_capacity;
                    response_metrics["winning_replay_capacity"] =
                        options.reinforcement_winning_replay_capacity;
                    response_metrics["rl_checkpoint_saved"] =
                        reinforcement_decision.checkpoint_saved;
                    response_metrics["rl_trainable_lstm"] =
                        rl_stats.trainable_lstm;
                    response_metrics["rl_recurrent_checkpoint_loaded"] =
                        rl_stats.recurrent_checkpoint_loaded;
                    response_metrics["rl_recurrent_checkpoint_migrated"] =
                        rl_stats.recurrent_checkpoint_migrated;
                    response_metrics["rl_trainable_resnet"] =
                        rl_stats.trainable_resnet;
                    response_metrics["rl_resnet_checkpoint_loaded"] =
                        rl_stats.resnet_checkpoint_loaded;
                    response_metrics["rl_resnet_training_updates"] =
                        rl_stats.resnet_training_updates;
                    response_metrics["rl_resnet_parameter_checksum"] =
                        rl_stats.resnet_parameter_checksum;
                    response_metrics["rl_training_device"] =
                        rl_stats.training_device;
                    response_metrics["rl_errors"] =
                        reinforcement_errors.load();
                    response_metrics["training_reset_pending"] =
                        training_reset_pending;
                    response_metrics["training_reset_pending_id"] =
                        pending_training_reset_request.has_value()
                            ? pending_training_reset_request
                                  ->at("reset_id")
                                  .get<std::uint64_t>()
                            : 0U;
                    response_metrics["training_reset_retries"] =
                        training_reset_retries.load();
                    response_metrics["repeated_frame_training_paused"] =
                        repeated_frame_paused;
                    response_metrics["repeated_frame_duration_ms"] =
                        repeated_frame_duration_ms.load();
                    response_metrics["repeated_frame_count"] =
                        repeated_frame_count.load();
                } else {
                    response["agent_mode"] =
                        std::string(
                            fighter_character_slug(player_character)
                        ) + "_mock_teacher";
                    response["control_enabled"] = true;
                    response["cancel_pending_sequence"] = false;
                }
                if (temporal_output.has_value()) {
                    auto& response_metrics = response["metrics"];
                    response_metrics["temporal_lstm_ms"] =
                        temporal_output->lstm.inference_ms;
                    response_metrics["temporal_hidden_l2"] =
                        temporal_output->hidden_l2;
                    response_metrics["temporal_sequence_size"] =
                        temporal_output->lstm.steps.size();
                    response_metrics["temporal_frame_id"] =
                        temporal_output->lstm.steps.back().frame_id;
                    response_metrics["temporal_decision_id"] =
                        temporal_output->lstm.steps.back().decision_id;
                    response_metrics["temporal_applied_decision_id"] =
                        temporal_output->lstm.steps.back()
                            .applied_decision_id;
                    response_metrics["temporal_applied_command_known"] =
                        temporal_output->lstm.steps.back()
                            .applied_command_known;
                    if (temporal_output->action_prediction.has_value()) {
                        const auto& prediction =
                            *temporal_output->action_prediction;
                        std::array<bool, 10> expected_state{};
                        std::array<bool, 10> predicted_state{};
                        for (int command : commands) {
                            if (command >= 0 && command < 10) {
                                expected_state[
                                    static_cast<std::size_t>(command)
                                ] = true;
                            }
                        }
                        for (int command : prediction.commands) {
                            if (command >= 0 && command < 10) {
                                predicted_state[
                                    static_cast<std::size_t>(command)
                                ] = true;
                            }
                        }
                        response_metrics["student_shadow_enabled"] = true;
                        response_metrics["student_shadow_commands"] =
                            prediction.commands;
                        response_metrics["student_shadow_probabilities"] =
                            prediction.probabilities;
                        response_metrics["student_shadow_active_probability"] =
                            prediction.active_probability;
                        response_metrics["student_shadow_action_active"] =
                            prediction.action_active;
                        response_metrics["student_shadow_exact_match"] =
                            expected_state == predicted_state;
                    } else {
                        response_metrics["student_shadow_enabled"] = false;
                    }
                }
                nlohmann::json metric_record = response;
                metric_record["connection_id"] = connection_id;
                if (!training_reset_pending) {
                    action_history.record_sent(
                        current_decision_id,
                        latest->frame_id,
                        commands,
                        now_ms,
                        timed_sequence_active
                    );
                    if (!writer.send_json(response)) {
                        connected.store(false);
                        shutdown(socket, SHUT_RDWR);
                        break;
                    }
                    metric_record["event"] = "decision";
                } else {
                    // A reset is an exclusive protocol operation. Sending an
                    // action first can hide the reset from clients that read
                    // one response for each submitted frame.
                    metric_record["event"] =
                        "decision_suppressed_for_training_reset";
                }
                metrics.write(metric_record);
                if (episode_summary.has_value()) {
                    writer.send_json(*episode_summary);
                    nlohmann::json summary_metric = *episode_summary;
                    summary_metric["event"] = "episode_summary";
                    summary_metric["connection_id"] = connection_id;
                    metrics.write(summary_metric);
                }
                if (evaluation_summary.has_value()) {
                    writer.send_json(*evaluation_summary);
                    nlohmann::json summary_metric = *evaluation_summary;
                    summary_metric["event"] = "evaluation_summary";
                    summary_metric["connection_id"] = connection_id;
                    metrics.write(summary_metric);
                }
                const auto telemetry_now = Clock::now();
                if (v21_session && reinforcement_agent != nullptr &&
                    telemetry_now >= next_telemetry_sample) {
                    const ReinforcementAgentStats stats =
                        reinforcement_agent->stats();
                    AgentMode telemetry_mode = AgentMode::training;
                    ExplorationMode telemetry_exploration =
                        ExplorationMode::automatic;
                    float telemetry_exploration_value =
                        options.reinforcement_new_matchup_epsilon_start;
                    ResetState telemetry_reset_state = ResetState::idle;
                    std::uint64_t telemetry_reset_id = 0;
                    bool telemetry_mode_pending = false;
                    {
                        std::lock_guard lock(protocol_session_mutex);
                        telemetry_mode = protocol_session.mode();
                        telemetry_exploration =
                            protocol_session.exploration_mode();
                        telemetry_exploration_value =
                            protocol_session.exploration_value();
                        telemetry_reset_state =
                            protocol_session.reset_state();
                        telemetry_reset_id =
                            protocol_session.active_reset_id();
                        telemetry_mode_pending =
                            protocol_session.pending_mode().has_value();
                    }
                    float epsilon_initial =
                        options.reinforcement_new_matchup_epsilon_start;
                    if (telemetry_exploration ==
                            ExplorationMode::manual_fixed ||
                        telemetry_exploration ==
                            ExplorationMode::automatic_from_value) {
                        epsilon_initial = telemetry_exploration_value;
                    }
                    const float epsilon_minimum =
                        options.reinforcement_epsilon_final;
                    const float epsilon_span =
                        epsilon_initial - epsilon_minimum;
                    const float epsilon_decay_progress =
                        telemetry_exploration ==
                                ExplorationMode::manual_fixed
                            ? 0.0f
                            : epsilon_span <= 0.0f
                                ? 1.0f
                                : std::clamp(
                                    (epsilon_initial - stats.epsilon) /
                                        epsilon_span,
                                    0.0f,
                                    1.0f
                                );
                    const auto evaluation_iterator = mode_statistics.find(
                        statistics_key(AgentMode::evaluation)
                    );
                    const ModeStatistics empty_evaluation_statistics;
                    const ModeStatistics& evaluation_statistics =
                        evaluation_iterator == mode_statistics.end()
                            ? empty_evaluation_statistics
                            : evaluation_iterator->second;
                    nlohmann::json telemetry{
                        {"protocol_version", kServerProtocolVersion},
                        {"request_id", telemetry_request_id++},
                        {"timestamp_ms", now_ms},
                        {"type", "telemetry"},
                        {"episode_id", fight.episode_id},
                        {"decision_id", current_decision_id},
                        {"agent_mode", agent_mode_slug(telemetry_mode)},
                        {"exploration_mode",
                         exploration_mode_slug(telemetry_exploration)},
                        {"encounter_type",
                         encounter_type_slug(encounter_configuration.type)},
                        {"encounter_event",
                         encounter_event_slug(latest_encounter_event)},
                        {"controlled_character",
                         fighter_character_slug(player_character)},
                        {"current_opponent", opponent_character_slug(
                            active_opponent_character.load()
                        )},
                        {"encounter_stage",
                         encounter_configuration.current_stage},
                        {"opponent_count",
                         encounter_configuration.opponent_count},
                        {"remaining_opponents",
                         encounter_configuration.opponent_count -
                            encounter_configuration.current_stage - 1U},
                        {"epsilon", stats.epsilon},
                        {"epsilon_initial", epsilon_initial},
                        {"epsilon_minimum", epsilon_minimum},
                        {"epsilon_decay_progress",
                         epsilon_decay_progress},
                        {"reward", response["reward_breakdown"]},
                        {"option_name", displayed_action_name(
                            reinforcement_decision.option_name,
                            reinforcement_decision.commands_mirrored
                        )},
                        {"option_name_base",
                         reinforcement_decision.option_name},
                        {"commands_mirrored",
                         reinforcement_decision.commands_mirrored},
                        {"selection_source",
                         reinforcement_decision.selection_source},
                        {"selected_q", reinforcement_decision.selected_q},
                        {"allowed_options",
                         reinforcement_decision.allowed_options},
                        {"training_loss", stats.last_training_loss},
                        {"predicted_q_mean", stats.last_predicted_q_mean},
                        {"td_target_mean", stats.last_td_target_mean},
                        {"bootstrap_mean", stats.last_bootstrap_mean},
                        {"q_target_gap", stats.last_q_target_gap},
                        {"abs_td_error", stats.last_abs_td_error},
                        {"critic_disagreement",
                         stats.last_critic_disagreement},
                        {"training_updates", stats.training_updates},
                        {"replay_size", stats.replay_size},
                        {"replay_capacity",
                         options.reinforcement_replay_capacity},
                        {"winning_replay_size", stats.winning_replay_size},
                        {"winning_replay_capacity",
                         options.reinforcement_winning_replay_capacity},
                        {"training_episodes",
                         stats.matchup_episodes_completed},
                        {"training_wins", stats.matchup_wins},
                        {"evaluation_session_id", evaluation_session_id},
                        {"evaluation_episodes",
                         evaluation_statistics.episodes},
                        {"evaluation_wins", evaluation_statistics.wins},
                        {"reset_state",
                         reset_state_slug(telemetry_reset_state)},
                        {"active_reset_id", telemetry_reset_id},
                        {"weights_frozen",
                         telemetry_mode == AgentMode::evaluation},
                        {"replay_frozen",
                         telemetry_mode == AgentMode::evaluation},
                        {"scheduler_frozen",
                         telemetry_mode == AgentMode::evaluation},
                        {"mode_change_pending", telemetry_mode_pending},
                    };
                    writer.send_json(telemetry);
                    telemetry["event"] = "telemetry";
                    telemetry["connection_id"] = connection_id;
                    metrics.write(telemetry);
                    nlohmann::json telemetry_sample{
                        {"protocol_version", kServerProtocolVersion},
                        {"request_id", telemetry_request_id++},
                        {"timestamp_ms", now_ms},
                        {"type", "telemetry_sample"},
                        {"episode_id", fight.episode_id},
                        {"decision_id", current_decision_id},
                        {"epsilon", reinforcement_decision.epsilon},
                        {"step_reward",
                         reinforcement_decision.reward_consumed},
                        {"episode_return",
                         reinforcement_decision.episode_return},
                        {"cumulative_damage_dealt",
                         episode_statistics.damage_dealt},
                        {"cumulative_damage_received",
                         episode_statistics.damage_received},
                        {"training_loss", stats.last_training_loss},
                        {"abs_td_error", stats.last_abs_td_error},
                        {"selected_q", reinforcement_decision.selected_q},
                        {"predicted_q_mean",
                         stats.last_predicted_q_mean},
                        {"replay_size", stats.replay_size},
                    };
                    writer.send_json(telemetry_sample);
                    telemetry_sample["event"] = "telemetry_sample";
                    telemetry_sample["connection_id"] = connection_id;
                    metrics.write(telemetry_sample);
                    next_telemetry_sample = telemetry_now +
                        std::chrono::milliseconds(100);
                }
                if (v21_session && reinforcement_step_ok &&
                    reinforcement_decision.checkpoint_saved) {
                    send_telemetry_marker(
                        "checkpoint_saved",
                        fight.episode_id,
                        static_cast<float>(
                            reinforcement_decision.training_updates
                        ),
                        reinforcement_agent->config()
                            .checkpoint_path.string()
                    );
                }
                bool created_training_reset = false;
                if (v21_session && pending_terminal_reset_reason.has_value() &&
                    !training_reset_trigger.has_value()) {
                    training_reset_trigger = TrainingResetTrigger{
                        .episode_id = fight.episode_id,
                        .frame_id = fight.frame_id,
                        .reason = std::string(
                            reset_reason_slug(*pending_terminal_reset_reason)
                        ),
                    };
                }
                if (
                    training_reset_trigger.has_value() &&
                    !pending_training_reset_request.has_value()
                ) {
                    const std::uint64_t reset_id =
                        last_training_reset_id.fetch_add(1U) + 1U;
                    const std::int64_t requested_at_ms =
                        epoch_milliseconds();
                    const bool compatibility_recovery =
                        training_reset_trigger->reason ==
                            "non_fight_timeout" ||
                        training_reset_trigger->reason ==
                            "repeated_frame_timeout";
                    // Preserve the original wire contract for clients that
                    // validate the legacy terminal-reset schema. The reset ID
                    // remains the idempotency key; episode_id is metadata and
                    // must be positive even before the first detected round.
                    const std::uint64_t compatible_episode_id =
                        std::max<std::uint64_t>(
                            1U,
                            training_reset_trigger->episode_id
                        );
                    const std::string compatible_reason =
                        compatibility_recovery
                            ? "round_lost"
                            : training_reset_trigger->reason;
                    if (v21_session) {
                        ResetReason reason =
                            pending_terminal_reset_reason.value_or(
                                terminal_reset_reason_from_trigger(
                                    training_reset_trigger->reason,
                                    compatibility_recovery
                                )
                            );
                        const bool terminal_committed =
                            fight.episode_ended && fight.terminal;
                        ResetTransition reset_transition;
                        {
                            std::lock_guard lock(protocol_session_mutex);
                            reset_transition = protocol_session.request_reset(
                                reason,
                                compatible_episode_id,
                                reset_id,
                                requested_at_ms,
                                terminal_committed
                            );
                        }
                        if (!reset_transition.accepted) {
                            writer.send_json({
                                {"protocol_version", kServerProtocolVersion},
                                {"request_id", telemetry_request_id++},
                                {"timestamp_ms", requested_at_ms},
                                {"type", "error"},
                                {"message", reset_transition.message},
                            });
                        } else {
                            ++reset_statistics.resets_requested;
                            if (reason == ResetReason::invalid_game_state) {
                                ++reset_statistics.invalid_state_resets;
                            }
                            if (reset_transition.episode_aborted) {
                                reinforcement_agent->suspend();
                            }
                            pending_training_reset_request = nlohmann::json{
                                {"protocol_version", kServerProtocolVersion},
                                {"request_id", telemetry_request_id++},
                                {"timestamp_ms", requested_at_ms},
                                {"type", "reset_episode"},
                                {"reset_id", reset_id},
                                {"episode_id", compatible_episode_id},
                                {"frame_id",
                                 training_reset_trigger->frame_id},
                                {"reason", reset_reason_slug(reason)},
                                {"state", "requested"},
                                {"load_save_state", true},
                                {"discard_inflight_action", true},
                                {"settle_delay_ms", 0},
                            };
                        }
                    } else {
                        pending_training_reset_request = nlohmann::json{
                            {"protocol_version", kActionProtocolVersion},
                            {"type", "training_reset_state"},
                            {"reset_id", reset_id},
                            {"episode_id", compatible_episode_id},
                            {"frame_id", training_reset_trigger->frame_id},
                            {"timestamp_ms", requested_at_ms},
                            {"reason", compatible_reason},
                            {"requested_operation",
                             "load_training_state"},
                            {"player_character",
                             fighter_character_slug(player_character)},
                            {"opponent_character",
                             opponent_character_slug(
                                 active_opponent_character.load()
                             )},
                            {"release_controls", true},
                            {"cancel_pending_sequence", true},
                            {"ack_required", true},
                        };
                    }
                    pending_training_reset_detected_reason =
                        training_reset_trigger->reason;
                    pending_training_reset_attempt = 0;
                    next_training_reset_retry = Clock::time_point::min();
                    created_training_reset = true;
                    if (v21_session &&
                        pending_training_reset_request.has_value()) {
                        send_telemetry_marker(
                            "reset_requested",
                            compatible_episode_id,
                            static_cast<float>(reset_id),
                            pending_training_reset_request->at("reason")
                                .get<std::string>()
                        );
                    }
                    pending_terminal_reset_reason.reset();
                }
                const auto reset_send_now = Clock::now();
                const bool reset_waiting_for_ack = [&] {
                    if (!v21_session) {
                        return true;
                    }
                    std::lock_guard lock(protocol_session_mutex);
                    const ResetState state = protocol_session.reset_state();
                    return state == ResetState::requested ||
                        state == ResetState::command_sent ||
                        state == ResetState::client_executing;
                }();
                if (
                    pending_training_reset_request.has_value() &&
                    reset_waiting_for_ack &&
                    (
                        created_training_reset ||
                        reset_send_now >= next_training_reset_retry
                    )
                ) {
                    nlohmann::json reset_request =
                        *pending_training_reset_request;
                    ++pending_training_reset_attempt;
                    const bool retry =
                        pending_training_reset_attempt > 1U;
                    reset_request["attempt"] =
                        pending_training_reset_attempt;
                    reset_request["retry"] = retry;
                    if (!writer.send_json(reset_request)) {
                        connected.store(false);
                        shutdown(socket, SHUT_RDWR);
                        break;
                    }
                    if (v21_session) {
                        std::lock_guard lock(protocol_session_mutex);
                        protocol_session.mark_reset_command_sent(
                            reset_request.at("reset_id")
                                .get<std::uint64_t>()
                        );
                    }
                    next_training_reset_retry =
                        reset_send_now + training_reset_retry_interval;
                    if (retry) {
                        training_reset_retries.fetch_add(1U);
                    } else {
                        training_reset_requests.fetch_add(1U);
                    }
                    nlohmann::json reset_metric = reset_request;
                    reset_metric["detected_reason"] =
                        pending_training_reset_detected_reason.value_or(
                            reset_request.at("reason").get<std::string>()
                        );
                    reset_metric["event"] =
                        retry
                            ? "training_reset_state_retried"
                            : "training_reset_state_requested";
                    reset_metric["connection_id"] = connection_id;
                    metrics.write(reset_metric);
                    std::cout
                        << "\n[TRAINING_RESET] "
                        << (retry ? "retry" : "request")
                        << " reset_id="
                        << reset_request.at("reset_id").get<std::uint64_t>()
                        << " connection=" << connection_id
                        << " episode="
                        << reset_request.at("episode_id").get<std::uint64_t>()
                        << " frame="
                        << reset_request.at("frame_id").get<std::uint64_t>()
                        << " reason="
                        << reset_request.at("reason").get<std::string>()
                        << " detected_reason="
                        << pending_training_reset_detected_reason.value_or(
                            reset_request.at("reason").get<std::string>()
                        )
                        << " attempt=" << pending_training_reset_attempt
                        << " operation=load_training_state\n\n"
                        << std::flush;
                }
                if (frame_stream_control_message.has_value()) {
                    if (!writer.send_json(*frame_stream_control_message)) {
                        connected.store(false);
                        shutdown(socket, SHUT_RDWR);
                        break;
                    }
                    nlohmann::json stream_metric =
                        *frame_stream_control_message;
                    stream_metric["event"] =
                        stream_metric.at("type").get<std::string>();
                    stream_metric["connection_id"] = connection_id;
                    metrics.write(stream_metric);
                }
                if (decision_id == 1 || decision_id % 20U == 0U) {
                    std::cout << "[DECISION] connection=" << connection_id
                              << " character="
                              << fighter_character_slug(player_character)
                              << " opponent="
                              << opponent_character_slug(
                                     active_opponent_character.load()
                                 )
                              << " decision=" << decision_id
                              << " frame_id=" << latest->frame_id
                              << " latency_ms=" << latency_ms
                              << " server_frame_age_ms=" << server_frame_age_ms
                              << " inference_ms=" << inference_ms
                              << " buffer=" << buffer.size()
                              << " dropped=" << buffer.dropped_total()
                              << " temporal_sequence="
                              << (
                                  temporal_output.has_value()
                                      ? temporal_output->lstm.steps.size()
                                      : 0U
                              )
                              << " temporal_lstm_ms="
                              << (
                                  temporal_output.has_value()
                                      ? temporal_output->lstm.inference_ms
                                      : 0.0
                              );
                    if (reinforcement_agent != nullptr) {
                        std::cout
                            << " fight=" << fight_phase_name(fight.phase)
                            << " hp=" << fight.player_health << '/'
                            << fight.opponent_health
                            << " reward="
                            << (
                                reinforcement_step_ok
                                    ? reinforcement_decision
                                          .reward_consumed
                                    : fight.reward +
                                          combat.shaping_reward
                            )
                            << " x=" << combat.player_x << '/'
                            << combat.opponent_x
                            << " distance=" << combat.distance
                            << " combo=" << combat.consecutive_hits
                            << " return="
                            << reinforcement_decision.episode_return
                            << " epsilon=" << reinforcement_decision.epsilon
                            << " replay=" << reinforcement_decision.replay_size
                            << " updates="
                            << reinforcement_decision.training_updates
                            << " repeated_frame_paused="
                            << (repeated_frame_paused ? 1 : 0)
                            << " repeated_frame_ms="
                            << repeated_frame_duration_ms.load()
                            << " option=\""
                            << displayed_action_name(
                                   reinforcement_decision.option_name,
                                   reinforcement_decision.commands_mirrored
                               ) << '"'
                            << " step="
                            << reinforcement_decision.option_step << '/'
                            << reinforcement_decision.option_steps;
                    } else {
                        std::cout
                            << " move=\"" << mock_decision.move_name << '"'
                            << " step=" << mock_decision.step_index << '/'
                            << mock_decision.step_count
                            << " phase=" << mock_decision.phase;
                    }
                    std::cout
                        << " commands=" << nlohmann::json(commands).dump()
                        << '\n';
                }
            }

            next_decision += options.decision_interval;
            const auto now = Clock::now();
            if (next_decision < now) {
                next_decision = now + options.decision_interval;
            }
        }
    });

    std::vector<std::uint8_t> fragmented_payload;
    std::uint8_t fragmented_opcode = 0;
    FightPhase last_logged_fight_phase = FightPhase::no_hud;
    const auto process_frame_envelope =
        [&](FrameEnvelope envelope, std::int64_t received_ms) {
            if (envelope.jpeg.size() > options.max_jpeg_bytes) {
                throw std::invalid_argument(
                    "JPEG exceeds configured size limit"
                );
            }
            received_frames.fetch_add(1);
            auto shared_envelope =
                std::make_shared<FrameEnvelope>(std::move(envelope));
            auto decoded = decode_frame(
                *shared_envelope,
                options.max_jpeg_bytes,
                received_ms
            );
            RepeatedFrameStatus repeated_frame_status;
            if (repeated_frame_detector != nullptr) {
                repeated_frame_status =
                    repeated_frame_detector->observe(
                        *decoded,
                        received_ms
                    );
                repeated_frame_training_paused.store(
                    repeated_frame_status.training_paused
                );
                repeated_frame_duration_ms.store(
                    repeated_frame_status.unchanged_duration_ms
                );
                repeated_frame_count.store(
                    repeated_frame_status.repeated_frames
                );
                if (repeated_frame_status.reset_requested) {
                    repeated_frame_reset_frame_id.store(
                        repeated_frame_status.frame_id
                    );
                    repeated_frame_reset_generation.fetch_add(1U);
                }
                if (
                    repeated_frame_status.pause_started ||
                    repeated_frame_status.recovered ||
                    repeated_frame_status.reset_requested
                ) {
                    const std::string event =
                        repeated_frame_status.reset_requested
                            ? "repeated_frame_reset_requested"
                            : repeated_frame_status.pause_started
                                ? "repeated_frame_training_paused"
                                : "repeated_frame_recovered";
                    metrics.write({
                        {"event", event},
                        {"connection_id", connection_id},
                        {"frame_id", repeated_frame_status.frame_id},
                        {"timestamp_ms", received_ms},
                        {"unchanged_duration_ms",
                         repeated_frame_status.unchanged_duration_ms},
                        {"repeated_frames",
                         repeated_frame_status.repeated_frames},
                        {"training_paused",
                         repeated_frame_status.training_paused},
                    });
                    std::cout
                        << "\n[REPEATED_FRAME] connection="
                        << connection_id
                        << " event=" << event
                        << " frame=" << repeated_frame_status.frame_id
                        << " unchanged_ms="
                        << repeated_frame_status.unchanged_duration_ms
                        << " repeated_frames="
                        << repeated_frame_status.repeated_frames
                        << " training_paused="
                        << (
                            repeated_frame_status.training_paused
                                ? 1
                                : 0
                        )
                        << "\n\n" << std::flush;
                }
            }
            const FramePushResult push_result = buffer.push(decoded);
            if (push_result == FramePushResult::rejected_stale) {
                writer.send_json({
                    {"protocol_version", kActionProtocolVersion},
                    {"type", "warning"},
                    {"message", "stale or duplicate frame discarded"},
                });
                return;
            }
            bool temporal_frame_allowed =
                !repeated_frame_status.training_paused;
            if (
                fight_tracker != nullptr &&
                !repeated_frame_status.training_paused
            ) {
                const FightControlSnapshot fight =
                    fight_tracker->observe(*decoded);
                CombatStateSnapshot combat;
                if (combat_tracker != nullptr) {
                    combat = combat_tracker->observe(
                        *decoded,
                        fight,
                        latest_applied_record(action_history)
                    );
                }
                temporal_frame_allowed = fight.action_allowed;
                if (fight.phase != last_logged_fight_phase ||
                    fight.reward != 0.0f ||
                    !fight.transition_reason.empty()) {
                    metrics.write({
                        {"event", "fight_state"},
                        {"connection_id", connection_id},
                        {"frame_id", fight.frame_id},
                        {"episode_id", fight.episode_id},
                        {"phase", fight_phase_name(fight.phase)},
                        {"hud_valid", fight.hud_valid},
                        {"action_allowed", fight.action_allowed},
                        {"player_health", fight.player_health},
                        {"opponent_health", fight.opponent_health},
                        {"damage_dealt", fight.damage_dealt},
                        {"damage_taken", fight.damage_taken},
                        {"reward", fight.reward},
                        {"terminal_reward", fight.terminal_reward},
                        {"hud_confidence", fight.hud_confidence},
                        {"transition_reason", fight.transition_reason},
                        {"combat_state_version", 2U},
                        {"combat_spatial_valid",
                         combat.spatial_valid},
                        {"combat_spatial_confidence",
                         combat.spatial_confidence},
                        {"player_x", combat.player_x},
                        {"opponent_x", combat.opponent_x},
                        {"fighter_distance", combat.distance},
                        {"player_on_ground",
                         combat.player_on_ground},
                        {"opponent_on_ground",
                         combat.opponent_on_ground},
                        {"player_defending",
                         combat.player_defending},
                        {"opponent_defending",
                         combat.opponent_defending},
                        {"player_attacking",
                         combat.player_attacking},
                        {"opponent_attacking",
                         combat.opponent_attacking},
                        {"time_since_last_damage_ms",
                         combat.time_since_last_damage_ms},
                        {"consecutive_hits",
                         combat.consecutive_hits},
                        {"stationary_time_ms",
                         combat.stationary_time_ms},
                        {"corner_time_ms",
                         combat.corner_time_ms},
                        {"combat_shaping_reward",
                         combat.shaping_reward},
                        {"combo_reward", combat.combo_reward},
                        {"defense_reward", combat.defense_reward},
                        {"idle_penalty", combat.idle_penalty},
                        {"corner_penalty", combat.corner_penalty},
                    });
                    last_logged_fight_phase = fight.phase;
                }
            }
            if (temporal_frame_allowed) {
                enqueue_temporal(std::move(decoded));
            }
            enqueue_persistence(
                std::move(shared_envelope), received_ms
            );
        };

    try {
        while (connected.load() && !stop_requested.load()) {
            WebSocketFrame frame;
            if (pending_first_frame.has_value()) {
                frame = std::move(*pending_first_frame);
                pending_first_frame.reset();
            } else {
                if (!read_websocket_frame(
                        reader,
                        frame,
                        maximum_payload,
                        stop_requested
                    )) {
                    break;
                }
            }
            if (frame.opcode == 0x8U) {
                writer.send_frame(0x8U, frame.payload);
                break;
            }
            if (frame.opcode == 0x9U) {
                if (!writer.send_frame(0xAU, frame.payload)) {
                    break;
                }
                continue;
            }
            if (frame.opcode == 0xAU) {
                continue;
            }

            std::uint8_t message_opcode = frame.opcode;
            std::vector<std::uint8_t> message_payload;
            if (frame.opcode == 0x0U) {
                if (fragmented_opcode == 0U) {
                    throw std::invalid_argument("unexpected WebSocket continuation frame");
                }
                if (fragmented_payload.size() + frame.payload.size() > maximum_payload) {
                    throw std::invalid_argument("fragmented WebSocket message is too large");
                }
                fragmented_payload.insert(
                    fragmented_payload.end(), frame.payload.begin(), frame.payload.end()
                );
                if (!frame.final) {
                    continue;
                }
                message_opcode = fragmented_opcode;
                message_payload = std::move(fragmented_payload);
                fragmented_payload.clear();
                fragmented_opcode = 0;
            } else if (frame.opcode == 0x1U || frame.opcode == 0x2U) {
                if (fragmented_opcode != 0U) {
                    throw std::invalid_argument("new data frame received before fragmented message ended");
                }
                if (!frame.final) {
                    fragmented_opcode = frame.opcode;
                    fragmented_payload = std::move(frame.payload);
                    continue;
                }
                message_payload = std::move(frame.payload);
            } else {
                throw std::invalid_argument("unsupported WebSocket opcode");
            }

            bool processing_frame_message = message_opcode == 0x2U;
            try {
                const std::int64_t received_ms = epoch_milliseconds();
                if (message_opcode == 0x1U) {
                    const std::string text(message_payload.begin(), message_payload.end());
                    ClientTextMessage client_message = parse_client_text_message(text);
                    if (std::holds_alternative<ActionAcknowledgement>(client_message)) {
                        const auto result = action_history.acknowledge(
                            std::get<ActionAcknowledgement>(client_message),
                            received_ms
                        );
                        if (!result.accepted) {
                            acknowledgements_rejected.fetch_add(1);
                            writer.send_json({
                                {"protocol_version", kActionProtocolVersion},
                                {"type", "warning"},
                                {"message", result.reason},
                            });
                            std::cerr << "[ACTION_ACK] connection=" << connection_id
                                      << " rejected: " << result.reason << '\n';
                        } else {
                            acknowledgements_received.fetch_add(1);
                            metrics.write({
                                {"protocol_version", kActionProtocolVersion},
                                {"event", "action_ack"},
                                {"connection_id", connection_id},
                                {"decision_id", result.applied.decision_id},
                                {"based_on_frame_id",
                                 result.applied.based_on_frame_id},
                                {"applied_commands", result.applied.commands},
                                {"commands_match", result.commands_match},
                                {"commands_advisory",
                                 result.commands_advisory},
                                {"generated_timestamp_ms",
                                 result.applied.generated_timestamp_ms},
                                {"applied_timestamp_ms",
                                 result.applied.applied_timestamp_ms},
                                {"acknowledgement_received_timestamp_ms",
                                 result.applied
                                     .acknowledgement_received_timestamp_ms},
                                {"server_roundtrip_ms",
                                 result.server_roundtrip_ms},
                            });
                            if (!result.commands_match &&
                                !result.commands_advisory) {
                                std::cerr
                                    << "[ACTION_ACK] connection=" << connection_id
                                    << " decision="
                                    << result.applied.decision_id
                                    << " applied state differs from sent state\n";
                            }
                        }
                        continue;
                    }
                    if (std::holds_alternative<
                            TrainingResetAcknowledgement
                        >(client_message)) {
                        const auto& acknowledgement = std::get<
                            TrainingResetAcknowledgement
                        >(client_message);
                        const std::uint64_t expected =
                            last_training_reset_id.load();
                        const std::uint64_t already_acknowledged =
                            last_training_reset_ack_id.load();
                        if (
                            expected == 0 ||
                            acknowledgement.reset_id != expected ||
                            acknowledgement.reset_id <=
                                already_acknowledged
                        ) {
                            training_reset_acknowledgements_rejected
                                .fetch_add(1U);
                            const std::string reason =
                                expected == 0
                                    ? "no training reset is pending"
                                    : acknowledgement.reset_id != expected
                                        ? "training reset ID does not match the pending request"
                                        : "training reset acknowledgement is duplicated";
                            writer.send_json({
                                {"protocol_version",
                                 kActionProtocolVersion},
                                {"type", "warning"},
                                {"message", reason},
                                {"expected_reset_id", expected},
                            });
                            std::cerr
                                << "[TRAINING_RESET_ACK] connection="
                                << connection_id
                                << " reset_id="
                                << acknowledgement.reset_id
                                << " rejected: " << reason << '\n';
                        } else {
                            last_training_reset_ack_id.store(
                                acknowledgement.reset_id
                            );
                            training_reset_acknowledgements.fetch_add(
                                1U
                            );
                            metrics.write({
                                {"protocol_version",
                                 kActionProtocolVersion},
                                {"event", "training_reset_ack"},
                                {"connection_id", connection_id},
                                {"reset_id",
                                 acknowledgement.reset_id},
                                {"success", acknowledgement.success},
                                {"completed_timestamp_ms",
                                 acknowledgement
                                     .completed_timestamp_ms},
                                {"acknowledgement_received_timestamp_ms",
                                 received_ms},
                                {"message",
                                 acknowledgement.message.value_or("")},
                            });
                            std::cout
                                << "\n[TRAINING_RESET_ACK] reset_id="
                                << acknowledgement.reset_id
                                << " connection=" << connection_id
                                << " success="
                                << (
                                    acknowledgement.success
                                        ? "true"
                                        : "false"
                                );
                            if (acknowledgement.message.has_value()) {
                                std::cout << " message=\""
                                          << *acknowledgement.message
                                          << '"';
                            }
                            std::cout << "\n\n" << std::flush;
                        }
                        continue;
                    }
                    if (std::holds_alternative<SessionConfiguration>(
                            client_message
                        )) {
                        writer.send_json({
                            {"protocol_version",
                             kActionProtocolVersion},
                            {"type", "session_config_error"},
                            {"message",
                             "session_config is accepted only before the first frame; reconnect to change character"},
                        });
                        continue;
                    }
                    if (
                        std::holds_alternative<UpdateEncounterStageRequest>(
                            client_message
                        ) ||
                        std::holds_alternative<EncounterEventRequest>(
                            client_message
                        ) ||
                        std::holds_alternative<SetAgentModeRequest>(
                            client_message
                        ) ||
                        std::holds_alternative<SetExplorationRequest>(
                            client_message
                        ) ||
                        std::holds_alternative<RequestReset>(client_message) ||
                        std::holds_alternative<ResetAck>(client_message) ||
                        std::holds_alternative<ResetReady>(client_message) ||
                        std::holds_alternative<ResetFailed>(client_message)
                    ) {
                        if (!v21_session) {
                            writer.send_json({
                                {"protocol_version", kActionProtocolVersion},
                                {"type", "error"},
                                {"timestamp_ms", received_ms},
                                {"message",
                                 "V2.1 control messages require configure_session"},
                            });
                        } else if (reinforcement_agent == nullptr ||
                                   fight_tracker == nullptr) {
                            writer.send_json({
                                {"protocol_version", kServerProtocolVersion},
                                {"timestamp_ms", received_ms},
                                {"type", "error"},
                                {"message",
                                 "V2.1 agent controls require reinforcement learning to be enabled"},
                            });
                        } else {
                            enqueue_control_message(std::move(client_message));
                        }
                        continue;
                    }
                    if (std::holds_alternative<ConfigureSessionRequest>(
                            client_message
                        )) {
                        writer.send_json({
                            {"protocol_version", kActionProtocolVersion},
                            {"type", "configure_session_error"},
                            {"timestamp_ms", received_ms},
                            {"message",
                             "configure_session is accepted only before the first frame; reconnect to change encounter"},
                        });
                        continue;
                    }
                    processing_frame_message = true;
                    FrameEnvelope envelope = std::move(
                        std::get<FrameEnvelope>(client_message)
                    );
                    process_frame_envelope(
                        std::move(envelope), received_ms
                    );
                    continue;
                }

                FrameEnvelope envelope = parse_binary_frame_envelope(message_payload);
                process_frame_envelope(
                    std::move(envelope), received_ms
                );
            } catch (const std::exception& error) {
                if (processing_frame_message) {
                    decode_errors.fetch_add(1);
                } else {
                    client_message_errors.fetch_add(1);
                }
                writer.send_json({
                    {"protocol_version", kActionProtocolVersion},
                    {"type", "error"},
                    {"message", error.what()},
                });
                std::cerr << (processing_frame_message ? "[FRAME]" : "[CLIENT_MESSAGE]")
                          << " connection=" << connection_id
                          << " rejected: " << error.what() << '\n';
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "[WEBSOCKET] connection=" << connection_id
                  << " protocol error: " << error.what() << '\n';
        send_close(writer, 1002, error.what());
    }

    connected.store(false);
    inference_thread.request_stop();
    workers_stopping.store(true);
    temporal_queue_changed.notify_all();
    persistence_queue_changed.notify_all();
    shutdown(socket, SHUT_RDWR);
    inference_thread.join();
    temporal_worker.join();
    persistence_worker.join();
    std::optional<ReinforcementAgentStats> reinforcement_stats;
    if (reinforcement_agent != nullptr) {
        try {
            reinforcement_agent->reset_episode(true);
            reinforcement_stats = reinforcement_agent->stats();
        } catch (const std::exception& error) {
            std::cerr << "[RL] final checkpoint failed: "
                      << error.what() << '\n';
        }
    }
    metrics.write({
        {"event", "disconnect"},
        {"connection_id", connection_id},
        {"player_character",
         fighter_character_slug(player_character)},
        {"opponent_character",
         opponent_character_slug(active_opponent_character.load())},
        {"timestamp_ms", epoch_milliseconds()},
        {"frames_received", received_frames.load()},
        {"frames_dropped", buffer.dropped_total()},
        {"decode_errors", decode_errors.load()},
        {"acknowledgements_received", acknowledgements_received.load()},
        {"acknowledgements_rejected", acknowledgements_rejected.load()},
        {"training_reset_requests", training_reset_requests.load()},
        {"training_reset_retries", training_reset_retries.load()},
        {"training_reset_acknowledgements",
         training_reset_acknowledgements.load()},
        {"training_reset_acknowledgements_rejected",
         training_reset_acknowledgements_rejected.load()},
        {"last_training_reset_id", last_training_reset_id.load()},
        {"client_message_errors", client_message_errors.load()},
        {"temporal_model_errors", temporal_model_errors.load()},
        {"temporal_frames_processed", temporal_frames_processed.load()},
        {"temporal_frames_overwritten", temporal_frames_overwritten.load()},
        {"persistence_frames_saved", persistence_frames_saved.load()},
        {"persistence_frames_dropped", persistence_frames_dropped.load()},
        {"persistence_errors", persistence_errors.load()},
        {"repeated_frame_training_paused",
         repeated_frame_training_paused.load()},
        {"repeated_frame_duration_ms",
         repeated_frame_duration_ms.load()},
        {"repeated_frame_count", repeated_frame_count.load()},
        {"last_applied_decision_id", action_history.last_applied_decision_id()},
        {"rl_enabled", reinforcement_agent != nullptr},
        {"rl_replay_size", reinforcement_stats.has_value()
            ? reinforcement_stats->replay_size : 0U},
        {"rl_training_updates", reinforcement_stats.has_value()
            ? reinforcement_stats->training_updates : 0U},
        {"rl_shared_replay_matchups",
         reinforcement_stats.has_value()
            ? reinforcement_stats->shared_replay_matchups : 0U},
        {"rl_episodes_completed", reinforcement_stats.has_value()
            ? reinforcement_stats->episodes_completed : 0U},
        {"rl_errors", reinforcement_errors.load()},
    });
    std::cout << "[DISCONNECT] connection=" << connection_id
              << " received=" << received_frames.load()
              << " dropped=" << buffer.dropped_total()
              << " decode_errors=" << decode_errors.load()
              << " acknowledgements=" << acknowledgements_received.load()
              << " rejected_acknowledgements="
              << acknowledgements_rejected.load()
              << " training_resets="
              << training_reset_requests.load()
              << " training_reset_retries="
              << training_reset_retries.load()
              << " training_reset_acks="
              << training_reset_acknowledgements.load()
              << " temporal_overwritten="
              << temporal_frames_overwritten.load()
              << " persistence_dropped="
              << persistence_frames_dropped.load()
              << "; waiting for a new client\n";
}

}  // namespace

WebSocketServer::WebSocketServer(WebSocketServerOptions options)
    : options_(std::move(options)) {
    if (options_.host.empty() || options_.port == 0 || options_.frame_buffer_capacity == 0 ||
        options_.observation_frames == 0 ||
        options_.applied_action_history_size == 0 ||
        options_.persistence_queue_capacity == 0 ||
        options_.max_jpeg_bytes == 0 ||
        options_.decision_interval.count() <= 0 || options_.failsafe_timeout.count() <= 0 ||
        options_.sequence_input_hold.count() <= 0 ||
        options_.sequence_neutral_hold.count() <= 0 ||
        options_.sequence_final_release.count() <= 0 ||
        options_.sequence_actionable_ticks == 0 ||
        options_.reinforcement_reward_log_interval.count() <= 0 ||
        options_.training_reset_non_fight_timeout.count() <= 0 ||
        options_.training_reset_round_intro_timeout <
            options_.training_reset_non_fight_timeout ||
        options_.reset_ack_timeout.count() <= 0 ||
        options_.reset_ready_timeout.count() <= 0 ||
        options_.frame_stream_stall_timeout.count() <= 0 ||
        options_.frame_stream_recovery_interval.count() <= 0 ||
        options_.repeated_frame_pause_timeout.count() <= 0 ||
        options_.repeated_frame_reset_timeout.count() <= 0 ||
        options_.debug_capture_interval.count() <= 0 ||
        options_.capture_directory.empty()) {
        throw std::invalid_argument("invalid WebSocket server configuration");
    }
    if (options_.collect_training_data &&
        options_.training_capture_directory.empty()) {
        throw std::invalid_argument(
            "training capture directory cannot be empty"
        );
    }
    if (options_.observation_frames > options_.frame_buffer_capacity) {
        throw std::invalid_argument("observation frame count cannot exceed buffer capacity");
    }
    if (
        options_.repeated_frame_reset_timeout <
            options_.repeated_frame_pause_timeout
    ) {
        throw std::invalid_argument(
            "repeated-frame reset timeout cannot precede pause timeout"
        );
    }
    if (options_.failsafe_timeout < options_.decision_interval * 2) {
        throw std::invalid_argument(
            "failsafe timeout must be at least two decision intervals"
        );
    }
    if (options_.temporal_model_enabled &&
        (options_.resnet50_checkpoint.empty() ||
         options_.temporal_visual_projection_size == 0 ||
         options_.temporal_hidden_size == 0 ||
         options_.temporal_device_id < 0)) {
        throw std::invalid_argument(
            "invalid temporal model configuration"
        );
    }
    if (options_.reinforcement_learning_enabled &&
        (!options_.temporal_model_enabled ||
         options_.reinforcement_checkpoint_directory.empty() ||
         (
             options_.reinforcement_backend != "cpp" &&
             options_.reinforcement_backend != "python"
         ) ||
                 (
                     options_.reinforcement_trainable_resnet &&
                     (
                         !options_.reinforcement_trainable_lstm ||
                         options_.reinforcement_resnet_learning_rate <= 0.0f ||
                 options_.reinforcement_resnet_batch_size == 0 ||
                 options_.reinforcement_resnet_update_interval == 0 ||
                 options_.reinforcement_resnet_replay_images == 0
             )
         ) ||
         (
             options_.reinforcement_trainable_lstm &&
             (
                 options_.reinforcement_backend == "python" &&
                 (
                     options_.recurrent_service_script.empty() ||
                     !std::filesystem::is_regular_file(
                         options_.recurrent_service_script
                     )
                 )
             )
         ) ||
         options_.reinforcement_network_hidden_size == 0 ||
         options_.reinforcement_replay_capacity == 0 ||
         options_.reinforcement_minimum_replay_size == 0 ||
         options_.reinforcement_batch_size == 0 ||
         options_.reinforcement_minimum_replay_size >
            options_.reinforcement_replay_capacity ||
         options_.reinforcement_batch_size >
            options_.reinforcement_replay_capacity ||
         options_.reinforcement_target_update_interval == 0 ||
         options_.reinforcement_epsilon_decay_selections == 0 ||
         options_.reinforcement_gamma <= 0.0f ||
         options_.reinforcement_gamma > 1.0f ||
         options_.reinforcement_q_value_limit <= 0.0f ||
         options_.reinforcement_terminal_replay_fraction < 0.0f ||
         options_.reinforcement_terminal_replay_fraction > 1.0f ||
         options_.reinforcement_hard_negative_replay_fraction < 0.0f ||
         options_.reinforcement_hard_negative_replay_fraction > 1.0f ||
         options_.reinforcement_winning_replay_fraction < 0.0f ||
         options_.reinforcement_winning_replay_fraction > 1.0f ||
         options_.reinforcement_winning_replay_min_fraction < 0.0f ||
         options_.reinforcement_winning_replay_min_fraction >
            options_.reinforcement_winning_replay_fraction ||
         options_.reinforcement_winning_replay_ramp_episodes == 0 ||
         options_.reinforcement_winning_replay_capacity == 0 ||
         options_.reinforcement_terminal_replay_fraction +
                 options_.reinforcement_hard_negative_replay_fraction +
                 options_.reinforcement_winning_replay_fraction >
             1.0f ||
         options_.reinforcement_hard_negative_q_threshold < 0.0f ||
         options_.reinforcement_n_step_options == 0 ||
         options_.reinforcement_learning_rate <= 0.0f ||
         options_.reinforcement_neutral_option_penalty < 0.0f ||
         options_.reinforcement_combo_max_distance <= 0.0f ||
         options_.reinforcement_combo_max_distance > 1.0f ||
         options_.reinforcement_close_attack_max_distance <= 0.0f ||
         options_.reinforcement_close_attack_max_distance > 1.0f ||
         options_.reinforcement_throw_max_distance <= 0.0f ||
         options_.reinforcement_throw_max_distance >
             options_.reinforcement_close_attack_max_distance ||
         options_.reinforcement_sweep_max_distance <= 0.0f ||
         options_.reinforcement_sweep_max_distance > 1.0f ||
         options_.reinforcement_projectile_preferred_min_distance <= 0.0f ||
         options_.reinforcement_projectile_preferred_min_distance > 1.0f ||
         options_.reinforcement_projectile_q_preference_margin < 0.0f ||
         options_.reinforcement_disable_cloud_min_distance < 0.0f ||
         options_.reinforcement_disable_cloud_max_distance <=
             options_.reinforcement_disable_cloud_min_distance ||
         options_.reinforcement_disable_cloud_max_distance > 1.0f ||
         options_.reinforcement_advance_min_distance < 0.0f ||
         options_.reinforcement_advance_min_distance > 1.0f ||
         options_.reinforcement_defense_max_distance <= 0.0f ||
         options_.reinforcement_defense_max_distance > 1.0f ||
         options_.reinforcement_disable_cloud_cooldown_options == 0 ||
         options_.reinforcement_throw_cooldown_options == 0 ||
         options_.reinforcement_sweep_cooldown_options == 0 ||
         options_.reinforcement_projectile_impact_credit_min_steps == 0 ||
         options_.reinforcement_projectile_impact_credit_max_steps <
             options_.reinforcement_projectile_impact_credit_min_steps ||
         options_.reinforcement_greedy_training_episode_interval == 0 ||
         options_.reinforcement_greedy_training_epsilon < 0.0f ||
         options_.reinforcement_greedy_training_epsilon > 1.0f ||
         options_.reinforcement_epsilon_start < 0.0f ||
         options_.reinforcement_epsilon_start > 1.0f ||
         options_.reinforcement_epsilon_final < 0.0f ||
         options_.reinforcement_epsilon_final >
            options_.reinforcement_epsilon_start ||
         options_.reinforcement_new_matchup_epsilon_start <
            options_.reinforcement_epsilon_final ||
         options_.reinforcement_new_matchup_epsilon_start > 1.0f ||
         options_.reinforcement_new_matchup_epsilon_decay_selections == 0 ||
         options_.reinforcement_scripted_exploration_probability < 0.0f ||
         options_.reinforcement_scripted_exploration_probability > 1.0f)) {
        throw std::invalid_argument(
            "RL requires the ResNet temporal path and valid shared-policy settings"
        );
    }
}

int WebSocketServer::run(const std::atomic_bool& stop_requested) {
    SocketHandle listening = create_listening_socket(options_);
    MetricsLogger metrics(options_.metrics_path);
    std::unique_ptr<TemporalAgent> temporal_agent;
    if (options_.temporal_model_enabled) {
        std::cout << "[TEMPORAL] loading "
                  << (
                      options_.reinforcement_learning_enabled &&
                              options_.reinforcement_trainable_resnet
                          ? "external trainable"
                          : "frozen"
                  )
                  << " ResNet50 checkpoint "
                  << options_.resnet50_checkpoint << '\n';
        temporal_agent = std::make_unique<TemporalAgent>(
            TemporalAgentConfig{
                .resnet50_checkpoint = options_.resnet50_checkpoint,
                .temporal_policy_checkpoint =
                    options_.temporal_policy_checkpoint,
                .sequence_capacity = options_.observation_frames,
                .visual_projection_size =
                    options_.temporal_visual_projection_size,
                .hidden_size = options_.temporal_hidden_size,
                .device_id = options_.temporal_device_id,
                .initialization_seed =
                    options_.random_seed.value_or(0x4c53544dU),
                .verify_resnet50_checkpoint = true,
                .frozen_lstm_inference_enabled =
                    !(
                        options_.reinforcement_learning_enabled &&
                        options_.reinforcement_trainable_lstm
                    ),
                .external_visual_encoder =
                    options_.reinforcement_learning_enabled &&
                    options_.reinforcement_trainable_resnet,
            }
        );
        const auto& report = temporal_agent->checkpoint_report();
        std::cout << "[TEMPORAL] ready checkpoint_epoch=" << report.epoch
                  << " best_accuracy=" << report.best_accuracy
                  << " tensors=" << report.tensor_count
                  << " sequence=" << options_.observation_frames
                  << " hidden=" << options_.temporal_hidden_size
                  << " device=" << options_.temporal_device_id << '\n';
        if (const auto& temporal_report =
                temporal_agent->temporal_checkpoint_report();
            temporal_report.has_value()) {
            std::cout << "[TEMPORAL] policy checkpoint="
                      << options_.temporal_policy_checkpoint
                      << " epoch=" << temporal_report->epoch
                      << " step=" << temporal_report->training_step
                      << " best_val_loss="
                      << temporal_report->best_validation_loss
                      << " best_score="
                      << temporal_report->best_selection_score
                      << " active_threshold="
                      << temporal_report->active_threshold
                      << " command_threshold="
                      << temporal_report->command_threshold
                      << " parameters="
                      << temporal_report->parameter_count
                      << " mode=student_shadow\n";
        } else {
            std::cout << "[TEMPORAL] no trained policy checkpoint; "
                         "deterministic from-scratch encoder\n";
        }
    }
    if (options_.reinforcement_learning_enabled) {
        std::cout
            << "[RL] shared multi-character semi-Markov clipped twin "
               "Double DQN enabled input="
            << (
                options_.temporal_hidden_size +
                kReinforcementContextFeatureCount
            )
            << " hidden=" << options_.reinforcement_network_hidden_size
            << " embeddings=16/16"
            << " combat_state=v2"
            << " training="
            << options_.reinforcement_training_enabled
            << " player_side="
            << (options_.player_is_left ? "left/P1" : "right/P2")
            << '\n'
            << "[RL] checkpoint directory: "
            << options_.reinforcement_checkpoint_directory << '\n';
    }
    std::uint64_t connection_id = 0;
    // The learner owns the replay and must outlive individual WebSocket
    // sessions. Connections are handled serially, so no additional locking is
    // required when changing the current matchup.
    std::unique_ptr<ReinforcementAgent> reinforcement_agent_storage;

    while (wait_for_client(listening.value, stop_requested)) {
        sockaddr_storage peer{};
        socklen_t peer_size = sizeof(peer);
        SocketHandle client(accept(
            listening.value, reinterpret_cast<sockaddr*>(&peer), &peer_size
        ));
        if (client.value < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "[ACCEPT] failed: " << std::strerror(errno) << '\n';
            continue;
        }
        configure_client_socket(client.value);
        SocketReader reader(client.value);
        std::string raw_headers;
        if (!reader.read_http_headers(raw_headers, stop_requested)) {
            continue;
        }

        try {
            const HttpRequest request = parse_http_request(raw_headers);
            if (request.method == "GET" && route_path(request.path) == "/health") {
                send_http_json(client.value, 200, "OK", {
                    {"status", "ok"},
                    {"websocket_path", "/ws"},
                    {"protocol_version", kActionProtocolVersion},
                    {"action_mode", kActionMode},
                    {"supported_action_modes", {
                        "state", "timed_sequence"
                    }},
                    {"timed_sequences_enabled",
                     options_.timed_sequences_enabled},
                    {"timed_sequence_version", 2},
                    {"timed_sequence_step_semantics",
                     "exact_controller_state"},
                    {"sequence_input_hold_ms",
                     options_.sequence_input_hold.count()},
                    {"sequence_neutral_hold_ms",
                     options_.sequence_neutral_hold.count()},
                    {"sequence_final_release_ms",
                     options_.sequence_final_release.count()},
                    {"sequence_actionable_ticks",
                     options_.sequence_actionable_ticks},
                    {"decision_hz", 1000.0 / options_.decision_interval.count()},
                    {"control_interval_ms", options_.decision_interval.count()},
                    {"failsafe_timeout_ms", options_.failsafe_timeout.count()},
                    {"buffer_capacity", options_.frame_buffer_capacity},
                    {"observation_frames", options_.observation_frames},
                    {"applied_action_history_size",
                     options_.applied_action_history_size},
                    {"persistence_queue_capacity",
                     options_.persistence_queue_capacity},
                    {"default_character", "subzero"},
                    {"supported_characters",
                     supported_fighter_characters()},
                    {"supported_opponent_characters",
                     supported_opponent_characters()},
                    {"opponent_feature_count",
                     kOpponentCharacterFeatureCount},
                    {"opponent_vocabulary_size",
                     kOpponentCharacterFeatureCount},
                    {"default_facing",
                     options_.default_facing_right ? "right" : "left"},
                    {"save_frames", options_.save_frames},
                    {"debug_capture_interval_ms",
                     options_.debug_capture_interval.count()},
                    {"collect_training_data",
                     options_.collect_training_data},
                    {"temporal_model_enabled",
                     options_.temporal_model_enabled},
                    {"temporal_model_ready",
                     temporal_agent != nullptr},
                    {"temporal_hidden_size",
                     options_.temporal_hidden_size},
                    {"student_shadow_enabled",
                     temporal_agent != nullptr &&
                         temporal_agent
                             ->temporal_checkpoint_report()
                             .has_value()},
                    {"reinforcement_learning_enabled",
                     options_.reinforcement_learning_enabled},
                    {"reinforcement_training_enabled",
                     options_.reinforcement_training_enabled},
                    {"reinforcement_read_only",
                     options_.reinforcement_learning_enabled &&
                         !options_.reinforcement_training_enabled},
                    {"reinforcement_replay_retained_across_connections",
                     options_.reinforcement_learning_enabled},
                    {"reinforcement_replay_size",
                     reinforcement_agent_storage != nullptr
                         ? reinforcement_agent_storage->stats().replay_size
                         : 0U},
                    {"reinforcement_epsilon_start",
                     options_.reinforcement_epsilon_start},
                    {"reinforcement_epsilon_final",
                     options_.reinforcement_epsilon_final},
                    {"reinforcement_gamma",
                     options_.reinforcement_gamma},
                    {"reinforcement_q_value_limit",
                     options_.reinforcement_q_value_limit},
                    {"reinforcement_terminal_replay_fraction",
                     options_.reinforcement_terminal_replay_fraction},
                    {"reinforcement_hard_negative_replay_fraction",
                     options_.reinforcement_hard_negative_replay_fraction},
                    {"reinforcement_n_step_options",
                     options_.reinforcement_n_step_options},
                    {"reinforcement_conservative_bootstrap",
                     options_.reinforcement_conservative_bootstrap},
                    {"reinforcement_reward_log_interval_ms",
                     options_.reinforcement_reward_log_interval.count()},
                    {"combat_state_version", 2},
                    {"combat_context_features",
                     kReinforcementContextFeatureCount},
                    {"damage_dealt_reward_scale",
                     options_
                         .reinforcement_damage_dealt_reward_scale},
                    {"damage_taken_penalty_scale",
                     options_
                         .reinforcement_damage_taken_penalty_scale},
                    {"round_win_bonus",
                     options_.reinforcement_round_win_bonus},
                    {"round_loss_penalty",
                     options_.reinforcement_round_loss_penalty},
                    {"combo_reward_scale",
                     options_.reinforcement_combo_reward_scale},
                    {"defense_reward_per_second",
                     options_
                         .reinforcement_defense_reward_per_second},
                    {"idle_penalty_per_second",
                     options_
                         .reinforcement_idle_penalty_per_second},
                    {"corner_penalty_per_second",
                     options_
                         .reinforcement_corner_penalty_per_second},
                    {"neutral_option_penalty",
                     options_
                         .reinforcement_neutral_option_penalty},
                    {"neutral_suppressed_during_fight",
                     options_
                         .reinforcement_suppress_neutral_during_fight},
                    {"combo_max_distance",
                     options_.reinforcement_combo_max_distance},
                    {"close_attack_max_distance",
                     options_.reinforcement_close_attack_max_distance},
                    {"throw_max_distance",
                     options_.reinforcement_throw_max_distance},
                    {"sweep_max_distance",
                     options_.reinforcement_sweep_max_distance},
                    {"projectile_preferred_min_distance",
                     options_
                         .reinforcement_projectile_preferred_min_distance},
                    {"projectile_q_preference_margin",
                     options_
                         .reinforcement_projectile_q_preference_margin},
                    {"disable_cloud_cooldown_options",
                     options_
                         .reinforcement_disable_cloud_cooldown_options},
                    {"throw_cooldown_options",
                     options_.reinforcement_throw_cooldown_options},
                    {"sweep_cooldown_options",
                     options_.reinforcement_sweep_cooldown_options},
                    {"dynamic_facing",
                     options_.reinforcement_dynamic_facing},
                    {"melee_impact_credit_steps",
                     options_.reinforcement_melee_impact_credit_steps},
                    {"teleport_impact_credit_steps",
                     options_.reinforcement_teleport_impact_credit_steps},
                    {"projectile_impact_credit_min_steps",
                     options_
                         .reinforcement_projectile_impact_credit_min_steps},
                    {"projectile_impact_credit_max_steps",
                     options_
                         .reinforcement_projectile_impact_credit_max_steps},
                    {"training_reset_state_enabled",
                     options_.reinforcement_learning_enabled &&
                         options_.training_reset_state_enabled},
                    {"training_reset_non_fight_timeout_ms",
                     options_
                         .training_reset_non_fight_timeout.count()},
                    {"training_reset_round_intro_timeout_ms",
                     options_
                         .training_reset_round_intro_timeout.count()},
                    {"frame_stream_stall_timeout_ms",
                     options_.frame_stream_stall_timeout.count()},
                    {"frame_stream_recovery_interval_ms",
                     options_.frame_stream_recovery_interval.count()},
                    {"repeated_frame_pause_ms",
                     options_.repeated_frame_pause_timeout.count()},
                    {"repeated_frame_reset_ms",
                     options_.repeated_frame_reset_timeout.count()},
                    {"training_reset_trigger",
                     options_.reinforcement_training_enabled
                         ? "round_lost_or_non_fight_timeout"
                         : "round_terminal_or_non_fight_timeout"},
                    {"training_reset_ack_type",
                     "training_reset_ack"},
                    {"fight_gating_enabled",
                     options_.reinforcement_learning_enabled},
                    {"player_side",
                     options_.player_is_left ? "left/P1" : "right/P2"},
                    {"rl_checkpoint_scope", "shared_multi_character"},
                    {"rl_checkpoint_format", 4},
                    {"rl_character_embedding_size", 16},
                    {"rl_opponent_embedding_size", 16},
                    {"rl_checkpoint_directory",
                     options_.reinforcement_checkpoint_directory.string()},
                });
                continue;
            }
            if (!perform_websocket_upgrade(client.value, request, options_.api_token)) {
                continue;
            }
            ++connection_id;
            metrics.write({
                {"event", "connect"},
                {"connection_id", connection_id},
                {"timestamp_ms", epoch_milliseconds()},
            });
            std::cout << "[CONNECT] connection=" << connection_id
                      << " WebSocket upgraded; buffer=" << options_.frame_buffer_capacity
                      << " observation=" << options_.observation_frames
                      << " interval_ms=" << options_.decision_interval.count() << '\n';
            handle_websocket_connection(
                client.value,
                reader,
                options_,
                temporal_agent.get(),
                reinforcement_agent_storage,
                metrics,
                connection_id,
                stop_requested
            );
        } catch (const std::exception& error) {
            send_http_json(client.value, 400, "Bad Request", {{"error", error.what()}});
        }
    }

    std::cout << "GameAgent WebSocket server stopped\n";
    return 0;
}

}  // namespace gameagent
