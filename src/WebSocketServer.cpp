#include "WebSocketServer.h"

#include "ActionHistory.h"
#include "FrameBuffer.h"
#include "HealthBarDetector.h"
#include "MockAgent.h"
#include "ReinforcementAgent.h"
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
    const std::optional<AppliedActionRecord>& latest_applied
) {
    std::vector<float> state;
    state.reserve(temporal.lstm.hidden.size() + 15U);
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
    std::unique_ptr<ReinforcementAgent> reinforcement_agent_storage;
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
        try {
            reinforcement_agent_storage =
                std::make_unique<ReinforcementAgent>(
                    ReinforcementAgentConfig{
                        .input_size =
                            options.temporal_hidden_size + 15U,
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
                        .learning_rate =
                            options.reinforcement_learning_rate,
                        .epsilon_start =
                            options.reinforcement_epsilon_start,
                        .epsilon_final =
                            options.reinforcement_epsilon_final,
                        .epsilon_decay_selections =
                            options
                                .reinforcement_epsilon_decay_selections,
                        .scripted_exploration_probability =
                            options
                                .reinforcement_scripted_exploration_probability,
                        .random_seed =
                            options.random_seed.value_or(0x524c444eU),
                        .character = player_character,
                        .opponent_character = opponent_character,
                        .facing = facing,
                        .training_enabled =
                            options.reinforcement_training_enabled,
                        .checkpoint_writes_enabled =
                            options.reinforcement_training_enabled,
                        .resume = options.reinforcement_resume,
                        .save_replay = true,
                        .trainable_lstm =
                            options.reinforcement_trainable_lstm,
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
                        .recurrent_service_script =
                            options.recurrent_service_script,
                        .recurrent_checkpoint_path =
                            options
                                .reinforcement_checkpoint_directory /
                            "shared_recurrent_latest.pt",
                        .checkpoint_path = latest_checkpoint,
                        .best_checkpoint_path = best_checkpoint,
                    }
                );
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
                {"type", "session_config_ack"},
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
                {"training_reset_state_enabled",
                 reinforcement_agent != nullptr &&
                     options.training_reset_state_enabled},
                {"training_reset_trigger", "round_lost"},
                {"training_reset_ack_type", "training_reset_ack"},
                {"checkpoint_loaded",
                 reinforcement_agent != nullptr &&
                     reinforcement_agent->stats().checkpoint_loaded},
                {"trainable_lstm",
                 reinforcement_agent != nullptr &&
                     reinforcement_agent->stats().trainable_lstm},
                {"recurrent_checkpoint_loaded",
                 reinforcement_agent != nullptr &&
                     reinforcement_agent
                         ->stats()
                         .recurrent_checkpoint_loaded},
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
        {"trainable_lstm",
         reinforcement_agent != nullptr &&
             reinforcement_agent->stats().trainable_lstm},
        {"rl_training_device",
         reinforcement_agent != nullptr
             ? reinforcement_agent->stats().training_device
             : "none"},
        {"training_reset_state_enabled",
         reinforcement_agent != nullptr &&
             options.training_reset_state_enabled},
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
        << '\n';

    CircularFrameBuffer buffer(options.frame_buffer_capacity);
    std::unique_ptr<FightStateTracker> fight_tracker;
    std::unique_ptr<TrainingResetTracker> training_reset_tracker;
    if (reinforcement_agent != nullptr) {
        fight_tracker = std::make_unique<FightStateTracker>(FightStateConfig{
            .player_side = player_is_left
                ? PlayerSide::left
                : PlayerSide::right,
            .maximum_action_frame_age_ms =
                std::max<std::int64_t>(
                    250,
                    options.decision_interval.count() * 3
                ),
        });
        if (options.training_reset_state_enabled) {
            training_reset_tracker =
                std::make_unique<TrainingResetTracker>();
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
    Clock::time_point next_debug_capture = Clock::time_point::min();
    std::atomic_bool workers_stopping{false};

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
                        opponent_character_slug(opponent_character)
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
        auto next_reward_log =
            Clock::now() + options.reinforcement_reward_log_interval;
        float reward_log_accumulator = 0.0f;
        float terminal_reward_log_accumulator = 0.0f;
        float damage_dealt_log_accumulator = 0.0f;
        float damage_taken_log_accumulator = 0.0f;
        std::uint64_t reward_log_decisions = 0;
        std::uint64_t reward_log_active_decisions = 0;
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
                FightControlSnapshot fight;
                std::optional<TrainingResetTrigger>
                    training_reset_trigger;
                if (fight_tracker != nullptr) {
                    fight = fight_tracker->consume(decision_tick_ms);
                    if (training_reset_tracker != nullptr) {
                        training_reset_trigger =
                            training_reset_tracker->observe(fight);
                    }
                    if (fight.episode_ended && temporal_agent != nullptr) {
                        temporal_agent->reset();
                    }
                }
                std::optional<TemporalAgentOutput> temporal_output;
                const bool temporal_control_allowed =
                    reinforcement_agent == nullptr || fight.action_allowed;
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
                            latest_applied
                        );
                    }
                    try {
                        reinforcement_decision = reinforcement_agent->step({
                            .state = reinforcement_state,
                            .temporal_visual_feature =
                                temporal_output.has_value()
                                    ? std::span<const float>(
                                          temporal_output
                                              ->visual_feature
                                      )
                                    : std::span<const float>{},
                            .temporal_controller_features =
                                temporal_output.has_value()
                                    ? std::span<const float>(
                                          temporal_output
                                              ->controller_features
                                      )
                                    : std::span<const float>{},
                            .reward = fight.reward,
                            .action_allowed =
                                fight.action_allowed &&
                                temporal_output.has_value(),
                            .episode_started = fight.episode_started,
                            .episode_ended = fight.episode_ended,
                            .terminal = fight.terminal,
                            .episode_id = fight.episode_id,
                            .frame_id = fight.frame_id,
                            .decision_id = current_decision_id,
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
                } else {
                    mock_decision = agent.infer(observation, applied_actions);
                    commands = mock_decision.commands;
                }
                if (reinforcement_agent != nullptr) {
                    reward_log_accumulator += fight.reward;
                    terminal_reward_log_accumulator +=
                        fight.terminal_reward;
                    damage_dealt_log_accumulator += fight.damage_dealt;
                    damage_taken_log_accumulator += fight.damage_taken;
                    ++reward_log_decisions;
                    reward_log_active_decisions +=
                        fight.action_allowed ? 1U : 0U;
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
                            << opponent_character_slug(opponent_character)
                            << " connection=" << connection_id
                            << " episode=" << fight.episode_id
                            << " phase=" << fight_phase_name(fight.phase)
                            << " interval_ms="
                            << options.reinforcement_reward_log_interval.count()
                            << " damage_dealt="
                            << damage_dealt_log_accumulator
                            << " damage_taken="
                            << damage_taken_log_accumulator
                            << " hp=" << fight.player_health << '/'
                            << fight.opponent_health
                            << " active_ticks="
                            << reward_log_active_decisions << '/'
                            << reward_log_decisions
                            << " epsilon=" << stats.epsilon
                            << " replay=" << stats.replay_size
                            << " matchups="
                            << stats.shared_replay_matchups
                            << " updates=" << stats.training_updates
                            << " loss=" << stats.last_training_loss
                            << " option=\""
                            << reinforcement_decision.option_name << '"'
                            << "\n\n" << std::flush;
                        reward_log_accumulator = 0.0f;
                        terminal_reward_log_accumulator = 0.0f;
                        damage_dealt_log_accumulator = 0.0f;
                        damage_taken_log_accumulator = 0.0f;
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
                std::vector<std::uint64_t> frame_ids;
                frame_ids.reserve(observation.size());
                for (const auto& frame : observation) {
                    frame_ids.push_back(frame->frame_id);
                }

                nlohmann::json response{
                    {"protocol_version", kActionProtocolVersion},
                    {"type", "action"},
                    {"action_mode", kActionMode},
                    {"player_character",
                     fighter_character_slug(player_character)},
                    {"opponent_character",
                     opponent_character_slug(opponent_character)},
                    {"opponent_character_name",
                     opponent_character_name(opponent_character)},
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
                if (reinforcement_agent != nullptr) {
                    const ReinforcementAgentStats rl_stats =
                        reinforcement_agent->stats();
                    response["agent_mode"] = "reinforcement_learning";
                    response["control_enabled"] =
                        fight.action_allowed && temporal_output.has_value() &&
                        reinforcement_step_ok;
                    response["cancel_pending_sequence"] =
                        !fight.action_allowed || !reinforcement_step_ok;
                    response["fight_phase"] = fight_phase_name(fight.phase);
                    response["episode_id"] = fight.episode_id;
                    response["episode_ended"] = fight.episode_ended;
                    response["terminal"] = fight.terminal;
                    response["player_health"] = fight.player_health;
                    response["opponent_health"] = fight.opponent_health;
                    response["reward"] = fight.reward;
                    response["terminal_reward"] =
                        fight.terminal_reward;
                    response["rl_option"] =
                        reinforcement_decision.option_name;
                    response["rl_option_index"] =
                        reinforcement_decision.option_index;
                    response["rl_option_step"] =
                        reinforcement_decision.option_step;
                    response["rl_option_steps"] =
                        reinforcement_decision.option_steps;
                    response["rl_selection_source"] =
                        reinforcement_decision.selection_source;
                    auto& response_metrics = response["metrics"];
                    response_metrics["hud_valid"] = fight.hud_valid;
                    response_metrics["hud_confidence"] =
                        fight.hud_confidence;
                    response_metrics["damage_dealt"] =
                        fight.damage_dealt;
                    response_metrics["damage_taken"] =
                        fight.damage_taken;
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
                    response_metrics["rl_training_updates"] =
                        reinforcement_decision.training_updates;
                    response_metrics["rl_training_loss"] =
                        reinforcement_decision.training_loss;
                    response_metrics["rl_training_performed"] =
                        reinforcement_decision.training_performed;
                    response_metrics["rl_checkpoint_saved"] =
                        reinforcement_decision.checkpoint_saved;
                    response_metrics["rl_trainable_lstm"] =
                        rl_stats.trainable_lstm;
                    response_metrics["rl_recurrent_checkpoint_loaded"] =
                        rl_stats.recurrent_checkpoint_loaded;
                    response_metrics["rl_training_device"] =
                        rl_stats.training_device;
                    response_metrics["rl_errors"] =
                        reinforcement_errors.load();
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
                action_history.record_sent(
                    current_decision_id,
                    latest->frame_id,
                    commands,
                    now_ms
                );
                if (!writer.send_json(response)) {
                    connected.store(false);
                    shutdown(socket, SHUT_RDWR);
                    break;
                }

                nlohmann::json metric_record = response;
                metric_record["event"] = "decision";
                metric_record["connection_id"] = connection_id;
                metrics.write(metric_record);
                if (training_reset_trigger.has_value()) {
                    const std::uint64_t reset_id =
                        last_training_reset_id.fetch_add(1U) + 1U;
                    const std::int64_t requested_at_ms =
                        epoch_milliseconds();
                    const nlohmann::json reset_request{
                        {"protocol_version", kActionProtocolVersion},
                        {"type", "training_reset_state"},
                        {"reset_id", reset_id},
                        {"episode_id",
                         training_reset_trigger->episode_id},
                        {"frame_id", training_reset_trigger->frame_id},
                        {"timestamp_ms", requested_at_ms},
                        {"reason", training_reset_trigger->reason},
                        {"requested_operation",
                         "load_training_state"},
                        {"player_character",
                         fighter_character_slug(player_character)},
                        {"opponent_character",
                         opponent_character_slug(opponent_character)},
                        {"release_controls", true},
                        {"cancel_pending_sequence", true},
                        {"ack_required", true},
                    };
                    if (!writer.send_json(reset_request)) {
                        connected.store(false);
                        shutdown(socket, SHUT_RDWR);
                        break;
                    }
                    training_reset_requests.fetch_add(1U);
                    nlohmann::json reset_metric = reset_request;
                    reset_metric["event"] =
                        "training_reset_state_requested";
                    reset_metric["connection_id"] = connection_id;
                    metrics.write(reset_metric);
                    std::cout
                        << "\n[TRAINING_RESET] request reset_id="
                        << reset_id
                        << " connection=" << connection_id
                        << " episode="
                        << training_reset_trigger->episode_id
                        << " frame="
                        << training_reset_trigger->frame_id
                        << " reason="
                        << training_reset_trigger->reason
                        << " operation=load_training_state\n\n"
                        << std::flush;
                }
                if (decision_id == 1 || decision_id % 20U == 0U) {
                    std::cout << "[DECISION] connection=" << connection_id
                              << " character="
                              << fighter_character_slug(player_character)
                              << " opponent="
                              << opponent_character_slug(opponent_character)
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
                            << " reward=" << fight.reward
                            << " return="
                            << reinforcement_decision.episode_return
                            << " epsilon=" << reinforcement_decision.epsilon
                            << " replay=" << reinforcement_decision.replay_size
                            << " updates="
                            << reinforcement_decision.training_updates
                            << " option=\""
                            << reinforcement_decision.option_name << '"'
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
            const FramePushResult push_result = buffer.push(decoded);
            if (push_result == FramePushResult::rejected_stale) {
                writer.send_json({
                    {"protocol_version", kActionProtocolVersion},
                    {"type", "warning"},
                    {"message", "stale or duplicate frame discarded"},
                });
                return;
            }
            bool temporal_frame_allowed = true;
            if (fight_tracker != nullptr) {
                const FightControlSnapshot fight =
                    fight_tracker->observe(*decoded);
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
                            if (!result.commands_match) {
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
         opponent_character_slug(opponent_character)},
        {"timestamp_ms", epoch_milliseconds()},
        {"frames_received", received_frames.load()},
        {"frames_dropped", buffer.dropped_total()},
        {"decode_errors", decode_errors.load()},
        {"acknowledgements_received", acknowledgements_received.load()},
        {"acknowledgements_rejected", acknowledgements_rejected.load()},
        {"training_reset_requests", training_reset_requests.load()},
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
        options_.reinforcement_reward_log_interval.count() <= 0 ||
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
             options_.reinforcement_trainable_lstm &&
             (
                 options_.recurrent_service_script.empty() ||
                 !std::filesystem::is_regular_file(
                     options_.recurrent_service_script
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
         options_.reinforcement_learning_rate <= 0.0f ||
         options_.reinforcement_epsilon_start < 0.0f ||
         options_.reinforcement_epsilon_start > 1.0f ||
         options_.reinforcement_epsilon_final < 0.0f ||
         options_.reinforcement_epsilon_final >
            options_.reinforcement_epsilon_start ||
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
        std::cout << "[TEMPORAL] loading frozen ResNet50 checkpoint "
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
            << "[RL] shared multi-character semi-Markov Double DQN enabled input="
            << (options_.temporal_hidden_size + 15U)
            << " hidden=" << options_.reinforcement_network_hidden_size
            << " embeddings=16/16"
            << " training="
            << options_.reinforcement_training_enabled
            << " player_side="
            << (options_.player_is_left ? "left/P1" : "right/P2")
            << '\n'
            << "[RL] checkpoint directory: "
            << options_.reinforcement_checkpoint_directory << '\n';
    }
    std::uint64_t connection_id = 0;

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
                    {"reinforcement_epsilon_start",
                     options_.reinforcement_epsilon_start},
                    {"reinforcement_epsilon_final",
                     options_.reinforcement_epsilon_final},
                    {"reinforcement_reward_log_interval_ms",
                     options_.reinforcement_reward_log_interval.count()},
                    {"training_reset_state_enabled",
                     options_.reinforcement_learning_enabled &&
                         options_.training_reset_state_enabled},
                    {"training_reset_trigger", "round_lost"},
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
