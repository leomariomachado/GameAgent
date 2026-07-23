cmake_minimum_required(VERSION 3.28)
project(GameAgent LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

set(
        GAMEAGENT_BASEAI_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/../BaseAI"
        CACHE PATH
        "Path to the BaseAI source tree"
)

if (NOT EXISTS "${GAMEAGENT_BASEAI_SOURCE_DIR}/CMakeLists.txt")
    message(
            FATAL_ERROR
            "BaseAI was not found at ${GAMEAGENT_BASEAI_SOURCE_DIR}. "
            "Set GAMEAGENT_BASEAI_SOURCE_DIR to the BaseAI source directory."
    )
endif()

add_subdirectory(
        "${GAMEAGENT_BASEAI_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/baseai"
)

add_library(
        GameAgentMockCore
        src/FrameBuffer.cpp
        src/HttpServer.cpp
        src/MockAgent.cpp
        src/MockProtocol.cpp
        src/StreamProtocol.cpp
        src/WebSocketServer.cpp
)
target_include_directories(
        GameAgentMockCore
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
target_link_libraries(GameAgentMockCore PUBLIC BaseAI::BaseAI Threads::Threads)

add_executable(GameAgent main.cpp)
target_link_libraries(GameAgent PRIVATE GameAgentMockCore)

include(CTest)
if (BUILD_TESTING)
    add_executable(
            test_gameagent_mock_protocol
            tests/test_mock_protocol.cpp
    )
    target_link_libraries(
            test_gameagent_mock_protocol
            PRIVATE GameAgentMockCore
    )
    add_test(
            NAME gameagent_mock_protocol
            COMMAND test_gameagent_mock_protocol
    )

    add_executable(
            test_gameagent_http_server
            tests/test_http_server.cpp
    )
    target_link_libraries(
            test_gameagent_http_server
            PRIVATE GameAgentMockCore
    )
    add_test(
            NAME gameagent_http_server
            COMMAND test_gameagent_http_server
    )

    add_executable(
            test_gameagent_stream_protocol
            tests/test_stream_protocol.cpp
    )
    target_link_libraries(
            test_gameagent_stream_protocol
            PRIVATE GameAgentMockCore
    )
    add_test(
            NAME gameagent_stream_protocol
            COMMAND test_gameagent_stream_protocol
    )

    add_executable(
            test_gameagent_frame_buffer
            tests/test_frame_buffer.cpp
    )
    target_link_libraries(
            test_gameagent_frame_buffer
            PRIVATE GameAgentMockCore
    )
    add_test(
            NAME gameagent_frame_buffer
            COMMAND test_gameagent_frame_buffer
    )

    add_executable(
            test_gameagent_websocket_server
            tests/test_websocket_server.cpp
    )
    target_link_libraries(
            test_gameagent_websocket_server
            PRIVATE GameAgentMockCore
    )
    add_test(
            NAME gameagent_websocket_server
            COMMAND test_gameagent_websocket_server
    )
endif()
