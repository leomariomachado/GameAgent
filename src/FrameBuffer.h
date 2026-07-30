#pragma once

#include <image/image.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace gameagent {

struct BufferedFrame {
    std::uint64_t frame_id = 0;
    std::int64_t client_timestamp_ms = 0;
    std::int64_t received_timestamp_ms = 0;
    std::size_t jpeg_size = 0;
    // Retained only in memory. The trainable visual backend consumes the
    // original compressed frame directly, avoiding an RGB -> JPEG round trip.
    std::vector<std::uint8_t> jpeg;
    base::data::image::Image image;
};

enum class FramePushResult {
    accepted,
    accepted_and_dropped_oldest,
    rejected_stale,
};

class CircularFrameBuffer {
public:
    explicit CircularFrameBuffer(std::size_t capacity);

    FramePushResult push(std::shared_ptr<const BufferedFrame> frame);
    [[nodiscard]] std::vector<std::shared_ptr<const BufferedFrame>> latest(
        std::size_t count
    ) const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t capacity() const;
    [[nodiscard]] std::uint64_t dropped_total() const;

private:
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<std::shared_ptr<const BufferedFrame>> frames_;
    std::uint64_t dropped_total_ = 0;
};

}  // namespace gameagent
