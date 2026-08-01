#include "RepeatedFrameDetector.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

gameagent::BufferedFrame make_frame(
    std::uint64_t frame_id,
    std::uint8_t value,
    int width = 2
) {
    base::data::image::Image image(width, 2, 3);
    for (auto& byte : image.bytes()) {
        byte = value;
    }
    return gameagent::BufferedFrame{
        .frame_id = frame_id,
        .image = std::move(image),
    };
}

}  // namespace

int main() {
    try {
        gameagent::RepeatedFrameDetector detector({
            .pause_after = std::chrono::milliseconds(2000),
            .reset_after = std::chrono::milliseconds(3000),
        });
        auto frame = make_frame(1, 17);
        auto status = detector.observe(frame, 1000);
        require(status.image_changed && !status.training_paused,
                "first frame did not initialize the detector");

        frame.frame_id = 2;
        status = detector.observe(frame, 2999);
        require(!status.training_paused && !status.reset_requested,
                "training paused before two seconds");

        frame.frame_id = 3;
        status = detector.observe(frame, 3000);
        require(status.training_paused && status.pause_started &&
                    !status.reset_requested &&
                    status.unchanged_duration_ms == 2000,
                "training did not pause at two seconds");

        frame.frame_id = 4;
        status = detector.observe(frame, 4000);
        require(status.training_paused && status.reset_requested &&
                    status.unchanged_duration_ms == 3000,
                "reset was not requested at three seconds");

        frame.frame_id = 5;
        status = detector.observe(frame, 5000);
        require(status.training_paused && !status.reset_requested,
                "unchanged streak emitted duplicate reset requests");

        auto changed = make_frame(6, 18);
        status = detector.observe(changed, 5001);
        require(status.image_changed && status.recovered &&
                    !status.training_paused,
                "changed RGB frame did not resume processing");

        changed.frame_id = 7;
        status = detector.observe(changed, 7001);
        require(status.training_paused && status.pause_started,
                "new unchanged streak did not restart pause timer");
        changed.frame_id = 8;
        status = detector.observe(changed, 8001);
        require(status.reset_requested,
                "new unchanged streak did not restart reset timer");

        auto resized = make_frame(9, 18, 3);
        status = detector.observe(resized, 8002);
        require(status.image_changed && status.recovered,
                "dimension change did not recover the detector");

        bool invalid_rejected = false;
        try {
            gameagent::RepeatedFrameDetector invalid({
                .pause_after = std::chrono::milliseconds(3000),
                .reset_after = std::chrono::milliseconds(2000),
            });
        } catch (const std::invalid_argument&) {
            invalid_rejected = true;
        }
        require(invalid_rejected, "invalid thresholds were accepted");

        std::cout
            << "[PASS] repeated-frame pause/recovery/reset test passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
