#include "StreamProtocol.h"
#include "WebSocketServer.h"

#include <json.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void send_all(int socket_fd, std::span<const std::uint8_t> bytes) {
    std::size_t sent = 0;
    while (sent < bytes.size()) {
        const ssize_t result = send(socket_fd, bytes.data() + sent, bytes.size() - sent, 0);
        if (result < 0 && errno == EINTR) {
            continue;
        }
        if (result <= 0) {
            throw std::runtime_error("could not send WebSocket test data");
        }
        sent += static_cast<std::size_t>(result);
    }
}

void send_text(int socket_fd, const std::string& text) {
    send_all(
        socket_fd,
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(text.data()), text.size()
        )
    );
}

std::vector<std::uint8_t> receive_exact(int socket_fd, std::size_t size) {
    std::vector<std::uint8_t> bytes(size);
    std::size_t received_total = 0;
    while (received_total < size) {
        const ssize_t received = recv(
            socket_fd, bytes.data() + received_total, size - received_total, 0
        );
        if (received < 0 && errno == EINTR) {
            continue;
        }
        if (received <= 0) {
            throw std::runtime_error("WebSocket test connection closed unexpectedly");
        }
        received_total += static_cast<std::size_t>(received);
    }
    return bytes;
}

int connect_with_retry(std::uint16_t port) {
    for (int attempt = 0; attempt < 100; ++attempt) {
        const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (socket_fd < 0) {
            throw std::runtime_error("could not create WebSocket test socket");
        }
        timeval timeout{.tv_sec = 3, .tv_usec = 0};
        setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

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
    throw std::runtime_error("could not connect to WebSocket test server");
}

std::string receive_http_headers(int socket_fd) {
    std::string response;
    char byte = 0;
    while (!response.ends_with("\r\n\r\n")) {
        const ssize_t received = recv(socket_fd, &byte, 1, 0);
        if (received <= 0) {
            throw std::runtime_error("could not receive WebSocket upgrade response");
        }
        response.push_back(byte);
        if (response.size() > 65536U) {
            throw std::runtime_error("WebSocket upgrade response is too large");
        }
    }
    return response;
}

int upgrade(std::uint16_t port, bool authorized) {
    const int socket_fd = connect_with_retry(port);
    std::string request =
        "GET /ws HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n";
    if (authorized) {
        request += "Authorization: Bearer test-secret\r\n";
    }
    request += "\r\n";
    send_text(socket_fd, request);
    const std::string response = receive_http_headers(socket_fd);
    if (authorized) {
        require(response.starts_with("HTTP/1.1 101 Switching Protocols\r\n"),
                "authorized WebSocket upgrade did not return 101");
        require(response.find("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") !=
                    std::string::npos,
                "WebSocket upgrade returned a wrong accept key");
    } else {
        require(response.starts_with("HTTP/1.1 401 Unauthorized\r\n"),
                "unauthorized WebSocket upgrade did not return 401");
    }
    return socket_fd;
}

void send_masked_frame(
    int socket_fd,
    std::uint8_t opcode,
    std::span<const std::uint8_t> payload
) {
    const std::array<std::uint8_t, 4> mask{0x12U, 0x34U, 0x56U, 0x78U};
    std::vector<std::uint8_t> frame;
    frame.push_back(static_cast<std::uint8_t>(0x80U | opcode));
    if (payload.size() <= 125U) {
        frame.push_back(static_cast<std::uint8_t>(0x80U | payload.size()));
    } else if (payload.size() <= 0xffffU) {
        frame.push_back(0x80U | 126U);
        frame.push_back(static_cast<std::uint8_t>((payload.size() >> 8U) & 0xffU));
        frame.push_back(static_cast<std::uint8_t>(payload.size() & 0xffU));
    } else {
        frame.push_back(0x80U | 127U);
        for (int shift = 56; shift >= 0; shift -= 8) {
            frame.push_back(static_cast<std::uint8_t>((payload.size() >> shift) & 0xffU));
        }
    }
    frame.insert(frame.end(), mask.begin(), mask.end());
    for (std::size_t index = 0; index < payload.size(); ++index) {
        frame.push_back(payload[index] ^ mask[index % mask.size()]);
    }
    send_all(socket_fd, frame);
}

nlohmann::json receive_json_frame(int socket_fd) {
    const auto header = receive_exact(socket_fd, 2);
    require(header[0] == 0x81U, "server response is not a final text frame");
    require((header[1] & 0x80U) == 0, "server response must not be masked");
    std::uint64_t size = header[1] & 0x7fU;
    if (size == 126U) {
        const auto extended = receive_exact(socket_fd, 2);
        size = (static_cast<std::uint64_t>(extended[0]) << 8U) | extended[1];
    } else if (size == 127U) {
        const auto extended = receive_exact(socket_fd, 8);
        size = 0;
        for (std::uint8_t byte : extended) {
            size = (size << 8U) | byte;
        }
    }
    const auto payload = receive_exact(socket_fd, static_cast<std::size_t>(size));
    return nlohmann::json::parse(payload.begin(), payload.end());
}

nlohmann::json send_one_frame_and_receive_action(int socket_fd, std::uint64_t frame_id) {
    constexpr std::string_view jpeg_base64 =
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAACAAIDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAAB//EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAH/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AIi2L3//Z";
    const auto jpeg = gameagent::decode_base64(jpeg_base64);
    const std::int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    const auto envelope = gameagent::make_binary_frame_envelope(frame_id, now_ms, jpeg);
    send_masked_frame(socket_fd, 0x2U, envelope);
    return receive_json_frame(socket_fd);
}

void send_action_acknowledgement(
    int socket_fd,
    const nlohmann::json& action
) {
    const nlohmann::json acknowledgement{
        {"protocol_version", 2},
        {"type", "action_ack"},
        {"decision_id", action.at("decision_id")},
        {"applied_commands", action.at("commands")},
        {"applied_timestamp_ms",
         std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch()
         ).count()},
    };
    const std::string text = acknowledgement.dump();
    send_masked_frame(
        socket_fd,
        0x1U,
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(text.data()), text.size()
        )
    );
}

nlohmann::json receive_action_with_acknowledgement_visible(
    int socket_fd,
    std::uint64_t acknowledged_decision_id
) {
    for (int attempt = 0; attempt < 10; ++attempt) {
        nlohmann::json action = receive_json_frame(socket_fd);
        if (action.value("type", "") != "action") {
            continue;
        }
        if (action["metrics"].value("last_applied_decision_id", 0U) ==
            acknowledged_decision_id) {
            return action;
        }
    }
    throw std::runtime_error(
        "confirmed action was not visible after ten server decisions"
    );
}

void close_websocket(int socket_fd) {
    const std::array<std::uint8_t, 2> normal_close{0x03U, 0xe8U};
    try {
        send_masked_frame(socket_fd, 0x8U, normal_close);
    } catch (...) {
    }
    close(socket_fd);
}

}  // namespace

int main() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("gameagent_websocket_test_" + std::to_string(getpid()) + '_' +
         std::to_string(unique));
    const std::uint16_t port = static_cast<std::uint16_t>(
        30000 + (static_cast<unsigned>(getpid()) % 20000U)
    );
    std::atomic_bool stop_requested{false};
    std::exception_ptr server_error;
    std::thread server_thread;
    int first_client = -1;
    int second_client = -1;

    try {
        gameagent::WebSocketServer server({
            .host = "127.0.0.1",
            .port = port,
            .frame_buffer_capacity = 4,
            .observation_frames = 2,
            .applied_action_history_size = 2,
            .max_jpeg_bytes = 1024U * 1024U,
            .decision_interval = std::chrono::milliseconds(20),
            .failsafe_timeout = std::chrono::milliseconds(100),
            .api_token = "test-secret",
            .random_seed = 12345U,
            .metrics_path = root / "metrics.jsonl",
            .capture_directory = root / "captures",
            .save_frames = true,
        });
        server_thread = std::thread([&] {
            try {
                server.run(stop_requested);
            } catch (...) {
                server_error = std::current_exception();
            }
        });

        const int unauthorized = upgrade(port, false);
        close(unauthorized);

        first_client = upgrade(port, true);
        const nlohmann::json first_action = send_one_frame_and_receive_action(first_client, 11U);
        require(first_action.value("type", "") == "action", "first response is not an action");
        require(first_action.value("protocol_version", 0) == 2,
                "action protocol version is not 2");
        require(first_action.value("action_mode", "") == "state",
                "action mode is not state");
        require(first_action.value("control_interval_ms", 0) == 20,
                "action control interval is incorrect");
        require(first_action.value("failsafe_timeout_ms", 0) == 100,
                "action failsafe timeout is incorrect");
        require(first_action.value("frame_id", 0U) == 11U, "first action used wrong frame");
        require(first_action.value("mock_move", "").size() > 0,
                "Sub-Zero move name was not returned");
        require(first_action.value("mock_step", 0U) == 1U,
                "Sub-Zero move did not start at step one");
        require(first_action.value("mock_steps", 0U) >= 5U,
                "Sub-Zero move length was not returned");
        require(first_action.value("mock_step_duration_ms", 0) == 20,
                "mock step duration does not match the decision interval");
        require(first_action["commands"].is_array(),
                "first action commands are not an array");
        for (const auto& command : first_action["commands"]) {
            require(
                command.is_number_integer() &&
                    command.get<int>() >= 0 &&
                    command.get<int>() <= 9,
                "Sub-Zero action contains an invalid command"
            );
        }
        const nlohmann::json second_frame_action =
            send_one_frame_and_receive_action(first_client, 12U);
        require(second_frame_action.value("type", "") == "action",
                "second frame response is not an action");
        send_action_acknowledgement(first_client, first_action);
        const nlohmann::json action_after_first_ack =
            receive_action_with_acknowledgement_visible(
                first_client,
                first_action["decision_id"].get<std::uint64_t>()
            );
        require(
            action_after_first_ack["metrics"].value(
                "last_applied_decision_id", 0U
            ) == first_action["decision_id"].get<std::uint64_t>(),
            "confirmed action was not exposed to the next agent observation"
        );
        require(
            action_after_first_ack["metrics"].value(
                "applied_action_history_size", 0U
            ) == 1U,
            "confirmed action history size was not exposed"
        );
        send_action_acknowledgement(first_client, action_after_first_ack);
        close_websocket(first_client);
        first_client = -1;

        second_client = upgrade(port, true);
        const nlohmann::json second_action = send_one_frame_and_receive_action(second_client, 1U);
        require(second_action.value("type", "") == "action", "reconnect response is not an action");
        require(second_action.value("frame_id", 0U) == 1U,
                "new connection did not start with a fresh buffer");
        send_action_acknowledgement(second_client, second_action);
        const nlohmann::json action_after_second_ack =
            receive_action_with_acknowledgement_visible(
                second_client,
                second_action["decision_id"].get<std::uint64_t>()
            );
        require(
            action_after_second_ack["metrics"].value(
                "last_applied_decision_id", 0U
            ) == second_action["decision_id"].get<std::uint64_t>(),
            "reconnected client acknowledgement was not retained"
        );
        send_action_acknowledgement(second_client, action_after_second_ack);
        close_websocket(second_client);
        second_client = -1;

        stop_requested.store(true);
        server_thread.join();
        if (server_error) {
            std::rethrow_exception(server_error);
        }
        require(std::filesystem::exists(root / "metrics.jsonl"),
                "WebSocket metrics file was not created");
        std::ifstream metrics_input(root / "metrics.jsonl");
        const std::string metrics_text{
            std::istreambuf_iterator<char>(metrics_input),
            std::istreambuf_iterator<char>()
        };
        require(metrics_text.find(R"("event":"action_ack")") != std::string::npos,
                "accepted action acknowledgement was not recorded");
        require(metrics_text.find(R"("commands_match":true)") != std::string::npos,
                "acknowledged applied commands did not match the sent state");
        const std::filesystem::path capture_session =
            root / "captures" / ("session_" + std::to_string(getpid()));
        require(
            std::filesystem::exists(capture_session / "connection_1" / "frame_11.jpg") &&
                std::filesystem::exists(capture_session / "connection_1" / "frames.jsonl"),
            "first connection frame and metadata were not saved"
        );
        require(
            !std::filesystem::exists(
                capture_session / "connection_1" / "frame_12.jpg"
            ),
            "debug mode saved more than one JPEG for the connection"
        );
        require(
            std::filesystem::exists(capture_session / "connection_2" / "frame_1.jpg"),
            "reconnected client frame was not saved"
        );

        std::filesystem::remove_all(root);
        std::cout << "[PASS] GameAgent WebSocket receive/infer/reconnect test passed\n";
        return 0;
    } catch (const std::exception& error) {
        if (first_client >= 0) {
            close(first_client);
        }
        if (second_client >= 0) {
            close(second_client);
        }
        stop_requested.store(true);
        if (server_thread.joinable()) {
            server_thread.join();
        }
        std::filesystem::remove_all(root);
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
