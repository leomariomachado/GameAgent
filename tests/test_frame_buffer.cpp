#include "FrameBuffer.h"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::shared_ptr<const gameagent::BufferedFrame> frame(std::uint64_t frame_id) {
    auto value = std::make_shared<gameagent::BufferedFrame>();
    value->frame_id = frame_id;
    return value;
}

}  // namespace

int main() {
    try {
        gameagent::CircularFrameBuffer buffer(3);
        require(buffer.push(frame(1)) == gameagent::FramePushResult::accepted,
                "first frame was not accepted");
        buffer.push(frame(2));
        buffer.push(frame(3));
        require(
            buffer.push(frame(4)) == gameagent::FramePushResult::accepted_and_dropped_oldest,
            "full buffer did not drop the oldest frame"
        );
        require(
            buffer.push(frame(4)) == gameagent::FramePushResult::rejected_stale,
            "duplicate frame was not rejected"
        );
        const auto latest = buffer.latest(8);
        require(latest.size() == 3, "buffer size is not bounded");
        require(latest[0]->frame_id == 2 && latest[1]->frame_id == 3 &&
                    latest[2]->frame_id == 4,
                "buffer did not retain the newest frames in order");
        require(buffer.dropped_total() == 2, "drop counter is incorrect");

        std::cout << "[PASS] GameAgent circular frame buffer test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
