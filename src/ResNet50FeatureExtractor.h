#pragma once

#include <image/image.h>
#include <models/resnet50/resnet50.cuh>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>
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

struct ResNet50NamedDeviceTensor {
    std::string name;
    float* data = nullptr;
    float* gradient = nullptr;
    std::size_t size = 0;
};

// ImageNet ResNet50 visual feature extractor. Inference remains batch-1; an
// independent instance can train stage 4 and synchronize only that stage.
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
        bool verify_checkpoint_copy = true,
        std::size_t batch_size = 1
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

    // Native stage-4 fine-tuning path. The model is allocated at the batch
    // size passed to the constructor. Missing rows are filled by repeating
    // the final image, and callers must provide zero feature gradients for
    // those rows during backward.
    const __half* extract_batch_device(
        std::span<const base::data::image::Image> images,
        bool stage4_training
    );

    void zero_stage4_grad();
    void backward_stage4(const float* device_feature_gradient);
    void sync_stage4_weights();
    void copy_stage4_from(ResNet50FeatureExtractor& source);

    [[nodiscard]]
    std::vector<ResNet50NamedDeviceTensor> stage4_state_tensors();

    [[nodiscard]]
    std::vector<ResNet50NamedDeviceTensor> stage4_parameters();

    // Downloads an already-computed device feature without running the
    // backbone a second time.
    std::vector<float> download_device_feature(
        const __half* device_feature
    );

    std::vector<float> download_device_features(
        const __half* device_features,
        std::size_t count
    );

    // Convenience path for the current GameAgent prototype and its tests.
    // The online GPU LSTM consumes extract_device() directly.
    std::vector<float> extract(
        const base::data::image::Image& image
    );

    [[nodiscard]] bool loaded() const;
    [[nodiscard]] int device_id() const;
    [[nodiscard]] cudaStream_t stream() const;
    [[nodiscard]] std::size_t batch_size() const;

private:
    void release() noexcept;
    void prepare_input(
        const base::data::image::Image& image,
        float* destination,
        std::size_t destination_count
    );

    int device_id_ = 0;
    bool verify_checkpoint_copy_ = true;
    std::size_t batch_size_ = 1;
    bool loaded_ = false;
    cudaStream_t stream_ = nullptr;
    float* device_input_float_ = nullptr;
    __half* device_input_half_ = nullptr;
    float* device_feature_float_ = nullptr;
    std::vector<float> host_input_;
    base::models::ResNet50 model_;
};

}  // namespace gameagent
