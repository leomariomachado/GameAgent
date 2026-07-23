#pragma once

#include <image/image.h>
#include <models/resnet50/resnet50.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gameagent {

struct ResNet50CheckpointReport {
    std::size_t tensor_count = 0;
    std::size_t value_count = 0;
    std::uint64_t epoch = 0;
    std::uint64_t global_step = 0;
    float best_accuracy = 0.0f;
    std::uint64_t checksum = 0;
    float verification_max_abs = 0.0f;
};

// Frozen ImageNet ResNet50 used only as a visual feature extractor.
//
// Input:
//   RGB8 image of any positive size.
// Preprocessing:
//   bilinear resize to 224x224, conversion to CHW [0,1], then ImageNet
//   channel normalization.
// Output:
//   the 2048 values produced by global average pooling, before the
//   ImageNet-1000 classifier. This is the per-frame visual vector intended
//   for the future temporal model.
class ResNet50FeatureExtractor {
public:
    static constexpr int kInputWidth = 224;
    static constexpr int kInputHeight = 224;
    static constexpr int kInputChannels = 3;
    static constexpr int kImageNetClasses = 1000;
    static constexpr int kFeatureSize =
        base::models::ResNet50::kOutputChannels;

    explicit ResNet50FeatureExtractor(
        int device_id = 0,
        bool verify_checkpoint_copy = true
    );
    ~ResNet50FeatureExtractor();

    ResNet50FeatureExtractor(const ResNet50FeatureExtractor&) = delete;
    ResNet50FeatureExtractor& operator=(const ResNet50FeatureExtractor&) = delete;
    ResNet50FeatureExtractor(ResNet50FeatureExtractor&&) = delete;
    ResNet50FeatureExtractor& operator=(ResNet50FeatureExtractor&&) = delete;

    ResNet50CheckpointReport load_checkpoint(const std::string& path);

    // Returns a device FP16 vector with kFeatureSize elements. The pointer is
    // owned by this extractor and is overwritten by the next extraction.
    const __half* extract_device(
        const base::data::image::Image& image
    );

    // Convenience path for the current GameAgent prototype and its tests.
    // A future GPU LSTM should consume extract_device() directly.
    std::vector<float> extract(
        const base::data::image::Image& image
    );

    [[nodiscard]] bool loaded() const;
    [[nodiscard]] int device_id() const;
    [[nodiscard]] cudaStream_t stream() const;

private:
    void release() noexcept;
    void prepare_input(const base::data::image::Image& image);

    int device_id_ = 0;
    bool verify_checkpoint_copy_ = true;
    bool loaded_ = false;
    cudaStream_t stream_ = nullptr;
    float* device_input_float_ = nullptr;
    __half* device_input_half_ = nullptr;
    float* device_feature_float_ = nullptr;
    std::vector<float> host_input_;
    base::models::ResNet50 model_;
};

}  // namespace gameagent
