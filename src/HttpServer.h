#pragma once

#include "MockProtocol.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace gameagent {

struct HttpServerOptions {
    std::string host = "127.0.0.1";
    std::uint16_t port = 8080;
    std::size_t max_body_bytes = 128U * 1024U * 1024U;
    std::optional<std::string> api_token;
};

class HttpServer {
public:
    explicit HttpServer(HttpServerOptions options);

    int run(MockProtocol& protocol, const std::atomic_bool& stop_requested) const;

private:
    HttpServerOptions options_;
};

}  // namespace gameagent
