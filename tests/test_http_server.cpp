#include "HttpServer.h"
#include "MockProtocol.h"

#include <json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

nlohmann::json make_request() {
    nlohmann::json frames = nlohmann::json::array();
    for (std::size_t index = 0; index < gameagent::kExpectedFrameCount; ++index) {
        frames.push_back({
            {"timestamp_ms", 1784740000000LL + static_cast<long long>(index * 100)},
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

int connect_with_retry(std::uint16_t port) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd < 0) {
            throw std::runtime_error("could not create test socket");
        }

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(port);
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        if (connect(
                socket_fd,
                reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)
            ) == 0) {
            return socket_fd;
        }
        close(socket_fd);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    throw std::runtime_error("could not connect to test HTTP server");
}

void send_all(int socket_fd, const std::string& data) {
    std::size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t result = send(socket_fd, data.data() + sent, data.size() - sent, 0);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            throw std::runtime_error("could not send HTTP test request");
        }
        sent += static_cast<std::size_t>(result);
    }
}

std::string receive_all(int socket_fd) {
    std::string response;
    char buffer[4096];
    while (true) {
        const ssize_t received = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received < 0) {
            throw std::runtime_error(
                "could not receive HTTP response: " + std::string(std::strerror(errno))
            );
        }
        if (received == 0) {
            break;
        }
        response.append(buffer, static_cast<std::size_t>(received));
    }
    return response;
}

std::string post_commands(
    std::uint16_t port,
    const std::string& body,
    const std::optional<std::string>& token
) {
    std::string request =
        "POST /commands HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Type: application/json\r\n";
    if (token.has_value()) {
        request += "Authorization: Bearer " + *token + "\r\n";
    }
    request += "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Connection: close\r\n\r\n" + body;

    const int client = connect_with_retry(port);
    send_all(client, request);
    const std::string response = receive_all(client);
    close(client);
    return response;
}

}  // namespace

int main() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() /
        ("gameagent_http_test_" + std::to_string(getpid()) + '_' + std::to_string(unique));
    const std::uint16_t port = static_cast<std::uint16_t>(
        20000 + (static_cast<unsigned>(getpid()) % 20000U)
    );

    std::atomic_bool stop_requested{false};
    std::exception_ptr server_error;
    std::thread server_thread;

    try {
        gameagent::MockProtocol protocol(root, 98765U);
        gameagent::HttpServer server({
            .host = "127.0.0.1",
            .port = port,
            .max_body_bytes = 1024U * 1024U,
            .api_token = "test-secret",
        });
        server_thread = std::thread([&] {
            try {
                server.run(protocol, stop_requested);
            } catch (...) {
                server_error = std::current_exception();
            }
        });

        const std::string body = make_request().dump();
        const std::string unauthorized = post_commands(port, body, std::nullopt);
        require(
            unauthorized.starts_with("HTTP/1.1 401 Unauthorized\r\n"),
            "missing API token should return 401"
        );

        const std::string response = post_commands(port, body, "test-secret");

        stop_requested.store(true);
        server_thread.join();
        if (server_error) {
            std::rethrow_exception(server_error);
        }

        require(
            response.starts_with("HTTP/1.1 200 OK\r\n"),
            "HTTP request should return 200 OK"
        );
        const std::size_t body_start = response.find("\r\n\r\n");
        require(body_start != std::string::npos, "HTTP response should contain a body");
        const nlohmann::json response_json = nlohmann::json::parse(
            response.substr(body_start + 4)
        );
        require(response_json.size() == 1, "response should contain only commands");
        require(response_json["commands"].is_array(), "commands should be an array");
        require(
            response_json["commands"].size() >= 3 &&
                response_json["commands"].size() <= 10,
            "response should contain 3..10 commands"
        );

        std::filesystem::remove_all(root);
        std::cout << "[PASS] GameAgent HTTP server test passed\n";
        return 0;
    } catch (const std::exception& error) {
        stop_requested.store(true);
        if (server_thread.joinable()) {
            server_thread.join();
        }
        std::filesystem::remove_all(root);
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
