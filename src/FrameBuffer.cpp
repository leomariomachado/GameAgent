#include "FrameBuffer.h"

#include <algorithm>
#include <stdexcept>

namespace gameagent {

CircularFrameBuffer::CircularFrameBuffer(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::invalid_argument("frame buffer capacity must be positive");
    }
}

FramePushResult CircularFrameBuffer::push(std::shared_ptr<const BufferedFrame> frame) {
    if (!frame) {
        throw std::invalid_argument("cannot push a null frame");
    }
    std::lock_guard lock(mutex_);
    if (!frames_.empty() && frame->frame_id <= frames_.back()->frame_id) {
        ++dropped_total_;
        return FramePushResult::rejected_stale;
    }

    FramePushResult result = FramePushResult::accepted;
    if (frames_.size() == capacity_) {
        frames_.pop_front();
        ++dropped_total_;
        result = FramePushResult::accepted_and_dropped_oldest;
    }
    frames_.push_back(std::move(frame));
    return result;
}

std::vector<std::shared_ptr<const BufferedFrame>> CircularFrameBuffer::latest(
    std::size_t count
) const {
    std::lock_guard lock(mutex_);
    const std::size_t selected = std::min(count, frames_.size());
    const auto begin = frames_.end() - static_cast<std::ptrdiff_t>(selected);
    return {begin, frames_.end()};
}

std::size_t CircularFrameBuffer::size() const {
    std::lock_guard lock(mutex_);
    return frames_.size();
}

std::size_t CircularFrameBuffer::capacity() const {
    return capacity_;
}

std::uint64_t CircularFrameBuffer::dropped_total() const {
    std::lock_guard lock(mutex_);
    return dropped_total_;
}

}  // namespace gameagent
