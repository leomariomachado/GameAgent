#pragma once

#include "FrameBuffer.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gameagent {

struct RepeatedFrameDetectorConfig {
    std::chrono::milliseconds pause_after{2000};
    std::chrono::milliseconds reset_after{3000};
};

struct RepeatedFrameStatus {
    bool image_changed = false;
    bool training_paused = false;
    bool pause_started = false;
    bool recovered = false;
    bool reset_requested = false;
    std::int64_t unchanged_duration_ms = 0;
    std::uint64_t frame_id = 0;
    std::uint64_t repeated_frames = 0;
};

// Detects byte-identical decoded RGB frames even when the client keeps
// incrementing frame_id. It pauses learning before requesting recovery so a
// frozen emulator cannot fill replay or consume the epsilon schedule.
class RepeatedFrameDetector {
public:
    explicit RepeatedFrameDetector(
        RepeatedFrameDetectorConfig config = {}
    );

    [[nodiscard]] RepeatedFrameStatus observe(
        const BufferedFrame& frame,
        std::int64_t now_ms
    );

    void reset();

private:
    RepeatedFrameDetectorConfig config_;
    std::vector<std::uint8_t> previous_rgb_;
    int previous_width_ = 0;
    int previous_height_ = 0;
    int previous_channels_ = 0;
    std::int64_t unchanged_since_ms_ = 0;
    std::uint64_t repeated_frames_ = 0;
    bool training_paused_ = false;
    bool reset_emitted_ = false;
};

}  // namespace gameagent
