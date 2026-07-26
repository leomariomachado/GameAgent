#include "WebSocketServer.h"

#include "ActionHistory.h"
#include "FrameBuffer.h"
#include "MockAgent.h"
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
    MockAgent& agent,
    TemporalAgent* temporal_agent,
    MetricsLogger& metrics,
    std::uint64_t connection_id,
    const std::atomic_bool& stop_requested
) {
    CircularFrameBuffer buffer(options.frame_buffer_capacity);
    agent.reset_sequence();
    if (temporal_agent != nullptr) {
        temporal_agent->reset();
    }
    ActionHistory action_history(
        std::max<std::size_t>(64U, options.applied_action_history_size * 8U)
    );
    ConnectionWriter writer(socket);
    std::atomic_bool connected{true};
    std::atomic<std::uint64_t> received_frames{0};
    std::atomic<std::uint64_t> decode_errors{0};
    std::atomic<std::uint64_t> acknowledgements_received{0};
    std::atomic<std::uint64_t> acknowledgements_rejected{0};
    std::atomic<std::uint64_t> client_message_errors{0};
    std::atomic<std::uint64_t> temporal_model_errors{0};
    std::atomic<std::uint64_t> temporal_frames_overwritten{0};
    std::atomic<std::uint64_t> temporal_frames_processed{0};
    std::atomic<std::uint64_t> persistence_frames_dropped{0};
    std::atomic<std::uint64_t> persistence_frames_saved{0};
    std::atomic<std::uint64_t> persistence_errors{0};
    bool debug_frame_saved = false;
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
            const bool debug_capture =
                options.save_frames && !debug_frame_saved;
            if (!training_capture && !debug_capture) {
                return;
            }
            if (debug_capture) {
                debug_frame_saved = true;
            }
            FramePersistenceJob job{
                .envelope = std::move(envelope),
                .received_timestamp_ms = received_timestamp_ms,
                .latest_applied_action =
                    latest_applied_record(action_history),
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
                std::optional<TemporalAgentOutput> temporal_output;
                if (temporal_agent != nullptr && temporal_agent->ready()) {
                    try {
                        temporal_output =
                            temporal_agent->advance_and_infer(
                                latest_applied_record(action_history),
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
                const MockAgentDecision mock_decision = agent.infer(
                    observation, applied_actions
                );
                const std::vector<int>& commands = mock_decision.commands;
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
                    {"commands", commands},
                    {"mock_move", mock_decision.move_name},
                    {"mock_step", mock_decision.step_index},
                    {"mock_steps", mock_decision.step_count},
                    {"curriculum_source", mock_decision.action_source},
                    {"curriculum_phase", mock_decision.phase},
                    {"curriculum_branch", mock_decision.branch},
                    {"freeze_visual_score",
                     mock_decision.freeze_visual_score},
                    {"freeze_visual_delta",
                     mock_decision.freeze_visual_delta},
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
                if (decision_id == 1 || decision_id % 20U == 0U) {
                    std::cout << "[DECISION] connection=" << connection_id
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
                              )
                              << " move=\"" << mock_decision.move_name << '"'
                              << " step=" << mock_decision.step_index << '/'
                              << mock_decision.step_count
                              << " phase=" << mock_decision.phase
                              << " branch=" << mock_decision.branch
                              << " commands=" << nlohmann::json(commands).dump() << '\n';
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
    const std::size_t maximum_payload = options.max_jpeg_bytes + kMaximumFragmentOverhead;
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
            enqueue_temporal(std::move(decoded));
            enqueue_persistence(
                std::move(shared_envelope), received_ms
            );
        };

    try {
        while (connected.load() && !stop_requested.load()) {
            WebSocketFrame frame;
            if (!read_websocket_frame(reader, frame, maximum_payload, stop_requested)) {
                break;
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
    metrics.write({
        {"event", "disconnect"},
        {"connection_id", connection_id},
        {"timestamp_ms", epoch_milliseconds()},
        {"frames_received", received_frames.load()},
        {"frames_dropped", buffer.dropped_total()},
        {"decode_errors", decode_errors.load()},
        {"acknowledgements_received", acknowledgements_received.load()},
        {"acknowledgements_rejected", acknowledgements_rejected.load()},
        {"client_message_errors", client_message_errors.load()},
        {"temporal_model_errors", temporal_model_errors.load()},
        {"temporal_frames_processed", temporal_frames_processed.load()},
        {"temporal_frames_overwritten", temporal_frames_overwritten.load()},
        {"persistence_frames_saved", persistence_frames_saved.load()},
        {"persistence_frames_dropped", persistence_frames_dropped.load()},
        {"persistence_errors", persistence_errors.load()},
        {"last_applied_decision_id", action_history.last_applied_decision_id()},
    });
    std::cout << "[DISCONNECT] connection=" << connection_id
              << " received=" << received_frames.load()
              << " dropped=" << buffer.dropped_total()
              << " decode_errors=" << decode_errors.load()
              << " acknowledgements=" << acknowledgements_received.load()
              << " rejected_acknowledgements="
              << acknowledgements_rejected.load()
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
}

int WebSocketServer::run(const std::atomic_bool& stop_requested) {
    SocketHandle listening = create_listening_socket(options_);
    MetricsLogger metrics(options_.metrics_path);
    MockAgent agent(
        options_.random_seed,
        options_.mock_facing_right ? FighterFacing::right : FighterFacing::left
    );
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
                         "teacher-only mode\n";
        }
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
                    {"mock_character", "Sub-Zero"},
                    {"mock_facing",
                     options_.mock_facing_right ? "right" : "left"},
                    {"save_frames", options_.save_frames},
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
                agent,
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
