#include "RepeatedFrameDetector.h"

#include <stdexcept>

namespace gameagent {

RepeatedFrameDetector::RepeatedFrameDetector(
    RepeatedFrameDetectorConfig config
) : config_(config) {
    if (
        config_.pause_after.count() <= 0 ||
        config_.reset_after < config_.pause_after
    ) {
        throw std::invalid_argument(
            "invalid repeated-frame detector configuration"
        );
    }
}

RepeatedFrameStatus RepeatedFrameDetector::observe(
    const BufferedFrame& frame,
    std::int64_t now_ms
) {
    if (frame.image.empty() || now_ms < 0) {
        throw std::invalid_argument(
            "repeated-frame detector requires a decoded image and timestamp"
        );
    }

    const auto& rgb = frame.image.bytes();
    const bool comparable =
        unchanged_since_ms_ != 0 &&
        frame.image.width() == previous_width_ &&
        frame.image.height() == previous_height_ &&
        frame.image.channels() == previous_channels_ &&
        rgb == previous_rgb_ &&
        now_ms >= unchanged_since_ms_;

    if (!comparable) {
        const bool recovered = training_paused_;
        previous_rgb_ = rgb;
        previous_width_ = frame.image.width();
        previous_height_ = frame.image.height();
        previous_channels_ = frame.image.channels();
        unchanged_since_ms_ = now_ms;
        repeated_frames_ = 1;
        training_paused_ = false;
        reset_emitted_ = false;
        return RepeatedFrameStatus{
            .image_changed = true,
            .training_paused = false,
            .pause_started = false,
            .recovered = recovered,
            .reset_requested = false,
            .unchanged_duration_ms = 0,
            .frame_id = frame.frame_id,
            .repeated_frames = repeated_frames_,
        };
    }

    ++repeated_frames_;
    const std::int64_t duration_ms = now_ms - unchanged_since_ms_;
    const bool pause_now = duration_ms >= config_.pause_after.count();
    const bool pause_started = pause_now && !training_paused_;
    training_paused_ = pause_now;
    const bool reset_requested =
        duration_ms >= config_.reset_after.count() && !reset_emitted_;
    if (reset_requested) {
        reset_emitted_ = true;
    }

    return RepeatedFrameStatus{
        .image_changed = false,
        .training_paused = training_paused_,
        .pause_started = pause_started,
        .recovered = false,
        .reset_requested = reset_requested,
        .unchanged_duration_ms = duration_ms,
        .frame_id = frame.frame_id,
        .repeated_frames = repeated_frames_,
    };
}

void RepeatedFrameDetector::reset() {
    previous_rgb_.clear();
    previous_width_ = 0;
    previous_height_ = 0;
    previous_channels_ = 0;
    unchanged_since_ms_ = 0;
    repeated_frames_ = 0;
    training_paused_ = false;
    reset_emitted_ = false;
}

}  // namespace gameagent
