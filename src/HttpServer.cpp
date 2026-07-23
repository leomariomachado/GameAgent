#include "HttpServer.h"

#include <json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cctype>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace gameagent {
namespace {

constexpr std::size_t kMaximumHeaderBytes = 64U * 1024U;

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
    std::string body;
};

struct HttpResponse {
    int status = 500;
    std::string body;
};

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

std::string status_reason(int status) {
    switch (status) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 411: return "Length Required";
        case 413: return "Payload Too Large";
        case 422: return "Unprocessable Content";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        default: return "Error";
    }
}

HttpResponse json_error(int status, std::string message) {
    return HttpResponse{
        .status = status,
        .body = nlohmann::json{{"error", std::move(message)}}.dump(),
    };
}

bool send_all(int socket, std::string_view data) {
    std::size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t result = send(socket, data.data() + sent, data.size() - sent, 0);
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

void send_response(int socket, const HttpResponse& response) {
    std::ostringstream headers;
    headers << "HTTP/1.1 " << response.status << ' ' << status_reason(response.status) << "\r\n"
            << "Content-Type: application/json; charset=utf-8\r\n"
            << "Content-Length: " << response.body.size() << "\r\n"
            << "Access-Control-Allow-Origin: *\r\n"
            << "Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key\r\n"
            << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            << "Connection: close\r\n\r\n";
    const std::string header_text = headers.str();
    send_all(socket, header_text);
    send_all(socket, response.body);
}

std::optional<std::size_t> parse_content_length(const std::string& value) {
    std::size_t result = 0;
    const char* begin = value.data();
    const char* end = value.data() + value.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        return std::nullopt;
    }
    return result;
}

std::variant<HttpRequest, HttpResponse> read_request(
    int socket,
    std::size_t max_body_bytes
) {
    std::string data;
    data.reserve(16U * 1024U);
    std::array<char, 16U * 1024U> buffer{};

    std::size_t header_end = std::string::npos;
    while ((header_end = data.find("\r\n\r\n")) == std::string::npos) {
        const ssize_t received = recv(socket, buffer.data(), buffer.size(), 0);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            return json_error(400, "failed to read HTTP headers");
        }
        if (received == 0) {
            return json_error(400, "connection closed before HTTP headers completed");
        }
        data.append(buffer.data(), static_cast<std::size_t>(received));
        if (data.size() > kMaximumHeaderBytes + max_body_bytes) {
            return json_error(413, "request is too large");
        }
        if (data.size() > kMaximumHeaderBytes &&
            data.find("\r\n\r\n") == std::string::npos) {
            return json_error(413, "HTTP headers are too large");
        }
    }

    HttpRequest request;
    std::istringstream header_stream(data.substr(0, header_end));
    std::string request_line;
    if (!std::getline(header_stream, request_line)) {
        return json_error(400, "missing HTTP request line");
    }
    if (!request_line.empty() && request_line.back() == '\r') {
        request_line.pop_back();
    }

    std::string http_version;
    std::istringstream request_line_stream(request_line);
    if (!(request_line_stream >> request.method >> request.path >> http_version) ||
        !http_version.starts_with("HTTP/")) {
        return json_error(400, "invalid HTTP request line");
    }

    std::string header_line;
    while (std::getline(header_stream, header_line)) {
        if (!header_line.empty() && header_line.back() == '\r') {
            header_line.pop_back();
        }
        const std::size_t colon = header_line.find(':');
        if (colon == std::string::npos) {
            return json_error(400, "invalid HTTP header");
        }
        request.headers[lowercase(trim(header_line.substr(0, colon)))] =
            trim(header_line.substr(colon + 1));
    }

    if (const auto transfer = request.headers.find("transfer-encoding");
        transfer != request.headers.end() && lowercase(transfer->second) != "identity") {
        return json_error(501, "chunked transfer encoding is not supported");
    }

    std::size_t content_length = 0;
    if (request.method == "POST") {
        const auto length_header = request.headers.find("content-length");
        if (length_header == request.headers.end()) {
            return json_error(411, "Content-Length is required");
        }
        const auto parsed_length = parse_content_length(length_header->second);
        if (!parsed_length.has_value()) {
            return json_error(400, "invalid Content-Length");
        }
        content_length = *parsed_length;
        if (content_length > max_body_bytes) {
            return json_error(413, "JSON body exceeds configured limit");
        }
    }

    const std::size_t body_start = header_end + 4;
    while (data.size() - body_start < content_length) {
        const std::size_t remaining = content_length - (data.size() - body_start);
        const ssize_t received = recv(
            socket,
            buffer.data(),
            std::min(buffer.size(), remaining),
            0
        );
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            return json_error(400, "failed to read HTTP body");
        }
        if (received == 0) {
            return json_error(400, "connection closed before HTTP body completed");
        }
        data.append(buffer.data(), static_cast<std::size_t>(received));
    }

    request.body = data.substr(body_start, content_length);
    return request;
}

bool constant_time_equals(std::string_view left, std::string_view right) {
    if (left.size() != right.size()) {
        return false;
    }
    unsigned char difference = 0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        difference |= static_cast<unsigned char>(left[index] ^ right[index]);
    }
    return difference == 0;
}

bool is_authorized(
    const HttpRequest& request,
    const std::optional<std::string>& api_token
) {
    if (!api_token.has_value()) {
        return true;
    }

    if (const auto api_key = request.headers.find("x-api-key");
        api_key != request.headers.end() &&
        constant_time_equals(api_key->second, *api_token)) {
        return true;
    }

    const auto authorization = request.headers.find("authorization");
    constexpr std::string_view bearer_prefix = "Bearer ";
    return authorization != request.headers.end() &&
        authorization->second.starts_with(bearer_prefix) &&
        constant_time_equals(
            std::string_view(authorization->second).substr(bearer_prefix.size()),
            *api_token
        );
}

HttpResponse route_request(
    const HttpRequest& request,
    MockProtocol& protocol,
    const std::optional<std::string>& api_token
) {
    if (request.method == "OPTIONS") {
        return HttpResponse{.status = 204, .body = {}};
    }
    if (request.method == "GET" && request.path == "/health") {
        return HttpResponse{
            .status = 200,
            .body = nlohmann::json{
                {"status", "ok"},
                {"expected_frames", kExpectedFrameCount},
                {"response_format", "commands"},
            }.dump(),
        };
    }
    if (request.method == "GET") {
        return json_error(404, "route not found");
    }
    if (request.method != "POST") {
        return json_error(405, "only POST requests are accepted for commands");
    }
    if (!is_authorized(request, api_token)) {
        return json_error(401, "missing or invalid API token");
    }

    ProtocolResponse protocol_response = protocol.process(request.body);
    if (protocol_response.http_status == 200) {
        std::cout << "[REQUEST] saved 30 frames in "
                  << protocol_response.capture_directory << '\n';
    } else {
        std::cerr << "[REQUEST] rejected with status "
                  << protocol_response.http_status << ": "
                  << protocol_response.body.dump() << '\n';
    }
    return HttpResponse{
        .status = protocol_response.http_status,
        .body = protocol_response.body.dump(),
    };
}

SocketHandle create_listening_socket(const HttpServerOptions& options) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;

    addrinfo* raw_addresses = nullptr;
    const std::string port = std::to_string(options.port);
    const int lookup_result = getaddrinfo(
        options.host.empty() ? nullptr : options.host.c_str(),
        port.c_str(),
        &hints,
        &raw_addresses
    );
    if (lookup_result != 0) {
        throw std::runtime_error(
            "could not resolve bind address: " + std::string(gai_strerror(lookup_result))
        );
    }

    struct AddressGuard {
        addrinfo* value;
        ~AddressGuard() { freeaddrinfo(value); }
    } addresses{raw_addresses};

    for (addrinfo* address = raw_addresses; address != nullptr; address = address->ai_next) {
        SocketHandle listening(socket(address->ai_family, address->ai_socktype, address->ai_protocol));
        if (listening.value < 0) {
            continue;
        }

        int reuse = 1;
        setsockopt(listening.value, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
        if (bind(listening.value, address->ai_addr, address->ai_addrlen) != 0) {
            continue;
        }
        if (listen(listening.value, 16) != 0) {
            continue;
        }
        return listening;
    }

    throw std::runtime_error(
        "could not bind to " + options.host + ':' + std::to_string(options.port) +
        ": " + std::strerror(errno)
    );
}

void configure_client_socket(int socket) {
    timeval timeout{};
    timeout.tv_sec = 30;
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

}  // namespace

HttpServer::HttpServer(HttpServerOptions options) : options_(std::move(options)) {
    if (options_.host.empty()) {
        throw std::invalid_argument("HTTP host cannot be empty");
    }
    if (options_.port == 0) {
        throw std::invalid_argument("HTTP port cannot be zero");
    }
    if (options_.max_body_bytes == 0) {
        throw std::invalid_argument("maximum HTTP body size cannot be zero");
    }
}

int HttpServer::run(MockProtocol& protocol, const std::atomic_bool& stop_requested) const {
    SocketHandle listening = create_listening_socket(options_);
    std::cout << "[HTTP] listening on " << options_.host << ':' << options_.port << '\n';

    while (!stop_requested.load()) {
        pollfd descriptor{
            .fd = listening.value,
            .events = POLLIN,
            .revents = 0,
        };
        const int poll_result = poll(&descriptor, 1, 500);
        if (poll_result < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw std::runtime_error("poll failed: " + std::string(std::strerror(errno)));
        }
        if (poll_result == 0 || (descriptor.revents & POLLIN) == 0) {
            continue;
        }

        SocketHandle client(accept(listening.value, nullptr, nullptr));
        if (client.value < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "[HTTP] accept failed: " << std::strerror(errno) << '\n';
            continue;
        }
        configure_client_socket(client.value);

        auto request_or_error = read_request(client.value, options_.max_body_bytes);
        if (std::holds_alternative<HttpResponse>(request_or_error)) {
            send_response(client.value, std::get<HttpResponse>(request_or_error));
            continue;
        }
        send_response(
            client.value,
            route_request(
                std::get<HttpRequest>(request_or_error),
                protocol,
                options_.api_token
            )
        );
    }

    std::cout << "[HTTP] shutdown requested\n";
    return 0;
}

}  // namespace gameagent
