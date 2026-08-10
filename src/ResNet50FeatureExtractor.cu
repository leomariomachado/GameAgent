#include "ResNet50FeatureExtractor.h"

#include <checkpoint/checkpoint.h>
#include <image/image_resize.h>
#include <global_avg_pool2d.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace gameagent {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

std::size_t convolution_weight_size(
    const base::ops::Conv2dParams& parameters
) {
    return
        static_cast<std::size_t>(parameters.out_channels) *
        static_cast<std::size_t>(parameters.in_channels) *
        static_cast<std::size_t>(parameters.kernel_h) *
        static_cast<std::size_t>(parameters.kernel_w);
}

void append_conv_bn_state(
    std::vector<ResNet50NamedDeviceTensor>& destination,
    const std::string& prefix,
    base::models::blocks::ConvBnRelu& layer
) {
    const std::size_t channels =
        static_cast<std::size_t>(layer.bn_params.channels);
    destination.push_back({
        prefix + ".weight",
        layer.weight,
        layer.grad_weight,
        convolution_weight_size(layer.conv_params),
    });
    destination.push_back({
        prefix + ".bias",
        layer.bias,
        layer.grad_bias,
        static_cast<std::size_t>(layer.conv_params.out_channels),
    });
    destination.push_back({
        prefix + ".gamma", layer.gamma, layer.grad_gamma, channels
    });
    destination.push_back({
        prefix + ".beta", layer.beta, layer.grad_beta, channels
    });
    destination.push_back({
        prefix + ".running_mean", layer.running_mean, nullptr, channels,
    });
    destination.push_back({
        prefix + ".running_var", layer.running_var, nullptr, channels,
    });
}

std::vector<ResNet50NamedDeviceTensor> model_state_tensors(
    base::models::ResNet50& model
) {
    std::vector<ResNet50NamedDeviceTensor> tensors;
    tensors.reserve(320);
    append_conv_bn_state(tensors, "stem", model.stem);

    for (int index = 0;
         index < base::models::ResNet50::kBlockCount;
         ++index) {
        auto& block = model.blocks[static_cast<std::size_t>(index)];
        const std::string prefix = "blocks." + std::to_string(index);
        append_conv_bn_state(tensors, prefix + ".conv1", block.conv1);
        append_conv_bn_state(tensors, prefix + ".conv2", block.conv2);
        append_conv_bn_state(tensors, prefix + ".conv3", block.conv3);
        if (block.has_projection) {
            append_conv_bn_state(
                tensors, prefix + ".projection", block.projection
            );
        }
    }

    tensors.push_back({
        "classifier.weight",
        model.classifier.weight,
        model.classifier.grad_weight,
        model.classifier.weight_size(),
    });
    tensors.push_back({
        "classifier.bias",
        model.classifier.bias,
        model.classifier.grad_bias,
        model.classifier.bias_size(),
    });
    return tensors;
}

std::uint64_t update_checksum(
    std::uint64_t checksum,
    const float* values,
    std::size_t count
) {
    constexpr std::uint64_t prime = 1099511628211ULL;
    const auto* bytes = reinterpret_cast<const unsigned char*>(values);
    const std::size_t byte_count = count * sizeof(float);
    for (std::size_t index = 0; index < byte_count; ++index) {
        checksum ^= bytes[index];
        checksum *= prime;
    }
    return checksum;
}

__global__ void float_to_half_kernel(
    const float* input,
    __half* output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2half(input[index]);
    }
}

__global__ void half_to_float_kernel(
    const __half* input,
    float* output,
    std::size_t count
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __half2float(input[index]);
    }
}

void normalize_imagenet_chw(std::vector<float>& values) {
    constexpr std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    constexpr std::array<float, 3> standard_deviation{
        0.229f, 0.224f, 0.225f
    };
    constexpr std::size_t plane =
        static_cast<std::size_t>(ResNet50FeatureExtractor::kInputWidth) *
        ResNet50FeatureExtractor::kInputHeight;

    if (values.size() != plane * mean.size()) {
        throw std::runtime_error("unexpected ResNet50 input size");
    }
    for (std::size_t channel = 0; channel < mean.size(); ++channel) {
        float* data = values.data() + channel * plane;
        for (std::size_t index = 0; index < plane; ++index) {
            data[index] =
                (data[index] - mean[channel]) / standard_deviation[channel];
        }
    }
}

}  // namespace

ResNet50FeatureExtractor::ResNet50FeatureExtractor(
    int device_id,
    bool verify_checkpoint_copy,
    std::size_t batch_size
)
    : device_id_(device_id),
      verify_checkpoint_copy_(verify_checkpoint_copy),
      batch_size_(batch_size) {
    if (device_id_ < 0 || batch_size_ == 0 ||
        batch_size_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("ResNet50 inference device must be non-negative");
    }

    try {
        check_cuda(cudaSetDevice(device_id_), "set ResNet50 inference device");
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "create ResNet50 inference stream"
        );
        model_.allocate(
            static_cast<int>(batch_size_),
            kInputChannels,
            kInputHeight,
            kInputWidth,
            kImageNetClasses,
            device_id_,
            stream_
        );
        model_.eval();

        constexpr std::size_t input_count =
            static_cast<std::size_t>(kInputChannels) *
            kInputHeight *
            kInputWidth;
        host_input_.resize(batch_size_ * input_count);
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_input_float_),
                host_input_.size() * sizeof(float)
            ),
            "allocate ResNet50 FP32 input"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_input_half_),
                host_input_.size() * sizeof(__half)
            ),
            "allocate ResNet50 FP16 input"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_feature_float_),
                batch_size_ * static_cast<std::size_t>(kFeatureSize) *
                    sizeof(float)
            ),
            "allocate ResNet50 FP32 feature output"
        );
        check_cuda(
            cudaStreamSynchronize(stream_),
            "synchronize ResNet50 inference allocation"
        );
    } catch (...) {
        release();
        throw;
    }
}

ResNet50FeatureExtractor::~ResNet50FeatureExtractor() {
    release();
}

void ResNet50FeatureExtractor::release() noexcept {
    cudaSetDevice(device_id_);
    model_.release();
    if (device_input_float_ != nullptr) {
        cudaFree(device_input_float_);
        device_input_float_ = nullptr;
    }
    if (device_input_half_ != nullptr) {
        cudaFree(device_input_half_);
        device_input_half_ = nullptr;
    }
    if (device_feature_float_ != nullptr) {
        cudaFree(device_feature_float_);
        device_feature_float_ = nullptr;
    }
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    loaded_ = false;
}

ResNet50CheckpointReport ResNet50FeatureExtractor::load_checkpoint(
    const std::string& path
) {
    if (path.empty()) {
        throw std::invalid_argument("ResNet50 checkpoint path is empty");
    }
    loaded_ = false;

    base::checkpoint::CheckpointState checkpoint;
    if (!base::checkpoint::load_checkpoint_metadata(path, checkpoint)) {
        throw std::runtime_error(
            "failed to read ResNet50 checkpoint metadata: " + path
        );
    }

    const auto destinations = model_state_tensors(model_);
    if (destinations.size() != 320 ||
        checkpoint.model.tensors.size() != destinations.size()) {
        throw std::runtime_error(
            "checkpoint does not contain the expected 320 ResNet50 tensors"
        );
    }

    std::unordered_map<std::string, std::size_t> source_indices;
    source_indices.reserve(checkpoint.model.tensors.size());
    for (std::size_t index = 0;
         index < checkpoint.model.tensors.size();
         ++index) {
        const auto& tensor = checkpoint.model.tensors[index];
        if (tensor.name.empty() ||
            !source_indices.emplace(tensor.name, index).second) {
            throw std::runtime_error(
                "duplicate or empty tensor in ResNet50 checkpoint"
            );
        }
    }

    std::unordered_set<std::string> destination_names;
    destination_names.reserve(destinations.size());
    std::vector<std::vector<float>> host(destinations.size());
    std::size_t value_count = 0;

    for (std::size_t index = 0; index < destinations.size(); ++index) {
        const auto& destination = destinations[index];
        if (destination.data == nullptr || destination.size == 0 ||
            !destination_names.insert(destination.name).second) {
            throw std::runtime_error(
                "invalid ResNet50 destination tensor: " + destination.name
            );
        }
        const auto found = source_indices.find(destination.name);
        if (found == source_indices.end()) {
            throw std::runtime_error(
                "checkpoint is missing ResNet50 tensor: " + destination.name
            );
        }
        auto& source = checkpoint.model.tensors[found->second];
        if (source.dtype != base::checkpoint::DType::Float32 ||
            source.num_elements != destination.size) {
            throw std::runtime_error(
                "checkpoint tensor shape or dtype mismatch: " +
                destination.name
            );
        }
        if (value_count >
            std::numeric_limits<std::size_t>::max() - destination.size) {
            throw std::runtime_error("ResNet50 checkpoint value count overflow");
        }
        value_count += destination.size;
        host[index].resize(destination.size);
        source.data = host[index].data();
    }

    if (destination_names.size() != source_indices.size()) {
        throw std::runtime_error(
            "checkpoint contains unexpected ResNet50 model tensors"
        );
    }
    if (!base::checkpoint::load_checkpoint_selective(path, checkpoint)) {
        throw std::runtime_error(
            "failed to read ResNet50 checkpoint tensor data: " + path
        );
    }

    std::uint64_t checksum = 1469598103934665603ULL;
    check_cuda(cudaSetDevice(device_id_), "select ResNet50 upload device");
    for (std::size_t index = 0; index < destinations.size(); ++index) {
        for (float value : host[index]) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "non-finite checkpoint value in: " +
                    destinations[index].name
                );
            }
        }
        checksum = update_checksum(
            checksum, host[index].data(), host[index].size()
        );
        check_cuda(
            cudaMemcpyAsync(
                destinations[index].data,
                host[index].data(),
                host[index].size() * sizeof(float),
                cudaMemcpyHostToDevice,
                stream_
            ),
            "upload ResNet50 checkpoint tensor"
        );
    }

    model_.sync_weight_half(stream_);
    model_.eval();
    check_cuda(
        cudaStreamSynchronize(stream_),
        "synchronize ResNet50 checkpoint upload"
    );

    float maximum_absolute_error = 0.0f;
    if (verify_checkpoint_copy_) {
        std::vector<float> downloaded;
        for (std::size_t index = 0; index < destinations.size(); ++index) {
            downloaded.resize(host[index].size());
            check_cuda(
                cudaMemcpyAsync(
                    downloaded.data(),
                    destinations[index].data,
                    downloaded.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream_
                ),
                "download ResNet50 checkpoint verification tensor"
            );
            check_cuda(
                cudaStreamSynchronize(stream_),
                "synchronize ResNet50 checkpoint verification"
            );
            for (std::size_t value_index = 0;
                 value_index < downloaded.size();
                 ++value_index) {
                maximum_absolute_error = std::max(
                    maximum_absolute_error,
                    std::abs(
                        downloaded[value_index] - host[index][value_index]
                    )
                );
            }
            if (std::memcmp(
                    downloaded.data(),
                    host[index].data(),
                    downloaded.size() * sizeof(float)
                ) != 0) {
                throw std::runtime_error(
                    "ResNet50 checkpoint device verification failed: " +
                    destinations[index].name
                );
            }
        }
    }

    loaded_ = true;
    return {
        .tensor_count = destinations.size(),
        .value_count = value_count,
        .epoch = checkpoint.epoch,
        .global_step = checkpoint.global_step,
        .best_accuracy = checkpoint.best_accuracy,
        .checksum = checksum,
        .verification_max_abs = maximum_absolute_error,
    };
}

void ResNet50FeatureExtractor::prepare_input(
    const base::data::image::Image& image,
    float* destination,
    std::size_t destination_count
) {
    if (image.empty() || image.channels() != kInputChannels) {
        throw std::invalid_argument(
            "ResNet50 inference requires a non-empty RGB image"
        );
    }

    if (image.width() == kInputWidth && image.height() == kInputHeight) {
        base::data::image::image_to_chw_float(
            image, destination, destination_count
        );
    } else {
        const auto resized = base::data::image::resize_bilinear(
            image, kInputWidth, kInputHeight
        );
        base::data::image::image_to_chw_float(
            resized, destination, destination_count
        );
    }
    std::vector<float> normalized(destination, destination + destination_count);
    normalize_imagenet_chw(normalized);
    std::copy(normalized.begin(), normalized.end(), destination);
}

const __half* ResNet50FeatureExtractor::extract_device(
    const base::data::image::Image& image
) {
    if (!loaded_) {
        throw std::runtime_error(
            "ResNet50 features requested before loading a checkpoint"
        );
    }

    return extract_batch_device(
        std::span<const base::data::image::Image>(&image, 1), false
    );
}

const __half* ResNet50FeatureExtractor::extract_batch_device(
    std::span<const base::data::image::Image> images,
    bool stage4_training
) {
    if (!loaded_) {
        throw std::runtime_error(
            "ResNet50 features requested before loading a checkpoint"
        );
    }
    if (images.empty() || images.size() > batch_size_) {
        throw std::invalid_argument("invalid ResNet50 extraction batch");
    }
    constexpr std::size_t single_input_count =
        static_cast<std::size_t>(kInputChannels) * kInputHeight * kInputWidth;
    for (std::size_t row = 0; row < images.size(); ++row) {
        prepare_input(
            images[row],
            host_input_.data() + row * single_input_count,
            single_input_count
        );
    }
    for (std::size_t row = images.size(); row < batch_size_; ++row) {
        std::copy_n(
            host_input_.data() + (images.size() - 1U) * single_input_count,
            single_input_count,
            host_input_.data() + row * single_input_count
        );
    }
    if (stage4_training) {
        model_.eval();
        for (int index = 13; index < base::models::ResNet50::kBlockCount;
             ++index) {
            model_.blocks[static_cast<std::size_t>(index)].train();
        }
    } else {
        model_.eval();
    }
    const std::size_t input_count = host_input_.size();
    check_cuda(
        cudaMemcpyAsync(
            device_input_float_,
            host_input_.data(),
            input_count * sizeof(float),
            cudaMemcpyHostToDevice,
            stream_
        ),
        "upload normalized ResNet50 input"
    );
    constexpr int block_size = 256;
    float_to_half_kernel<<<
        static_cast<int>((input_count + block_size - 1U) / block_size),
        block_size,
        0,
        stream_
    >>>(device_input_float_, device_input_half_, input_count);
    check_cuda(cudaGetLastError(), "convert ResNet50 input to FP16");

    model_.forward_backbone(device_input_half_, stream_);
    base::ops::global_avg_pool2d_forward(
        model_.block_outputs[base::models::ResNet50::kBlockCount - 1],
        model_.gap_output,
        model_.gap_params,
        stream_
    );
    return model_.gap_output;
}

void ResNet50FeatureExtractor::zero_stage4_grad() {
    for (int index = 13; index < base::models::ResNet50::kBlockCount;
         ++index) {
        model_.blocks[static_cast<std::size_t>(index)].zero_grad(stream_);
    }
}

void ResNet50FeatureExtractor::backward_stage4(
    const float* device_feature_gradient
) {
    if (device_feature_gradient == nullptr) {
        throw std::invalid_argument("stage4 feature gradient is null");
    }
    const std::size_t feature_count =
        batch_size_ * static_cast<std::size_t>(kFeatureSize);
    constexpr int block_size = 256;
    float_to_half_kernel<<<
        static_cast<int>((feature_count + block_size - 1U) / block_size),
        block_size, 0, stream_
    >>>(device_feature_gradient, model_.grad_gap_half, feature_count);
    check_cuda(cudaGetLastError(), "convert stage4 feature gradient");
    base::ops::global_avg_pool2d_backward(
        model_.grad_gap_half,
        model_.grad_feature_a,
        model_.gap_params,
        stream_
    );
    __half* current_gradient = model_.grad_feature_a;
    for (int index = base::models::ResNet50::kBlockCount - 1;
         index >= 13; --index) {
        __half* next_gradient = current_gradient == model_.grad_feature_a
            ? model_.grad_feature_b : model_.grad_feature_a;
        model_.blocks[static_cast<std::size_t>(index)].backward(
            model_.block_outputs[static_cast<std::size_t>(index - 1)],
            current_gradient,
            next_gradient,
            stream_
        );
        current_gradient = next_gradient;
    }
}

void ResNet50FeatureExtractor::sync_stage4_weights() {
    for (int index = 13; index < base::models::ResNet50::kBlockCount;
         ++index) {
        model_.blocks[static_cast<std::size_t>(index)].sync_weight_half(stream_);
    }
    model_.eval();
}

void ResNet50FeatureExtractor::copy_stage4_from(
    ResNet50FeatureExtractor& source
) {
    if (!loaded_ || !source.loaded_ || device_id_ != source.device_id_) {
        throw std::invalid_argument(
            "stage4 synchronization requires loaded models on one device"
        );
    }
    check_cuda(
        cudaStreamSynchronize(source.stream_),
        "synchronize source ResNet50 stage4"
    );
    auto destination = stage4_state_tensors();
    auto origin = source.stage4_state_tensors();
    if (destination.size() != origin.size()) {
        throw std::logic_error("ResNet50 stage4 state layout mismatch");
    }
    for (std::size_t index = 0; index < destination.size(); ++index) {
        if (destination[index].name != origin[index].name ||
            destination[index].size != origin[index].size) {
            throw std::logic_error("ResNet50 stage4 tensor mismatch");
        }
        check_cuda(cudaMemcpyAsync(
            destination[index].data,
            origin[index].data,
            destination[index].size * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream_
        ), "copy ResNet50 stage4 tensor");
    }
    sync_stage4_weights();
    check_cuda(
        cudaStreamSynchronize(stream_),
        "synchronize destination ResNet50 stage4"
    );
}

std::vector<ResNet50NamedDeviceTensor>
ResNet50FeatureExtractor::stage4_state_tensors() {
    std::vector<ResNet50NamedDeviceTensor> result;
    result.reserve(60);
    for (int index = 13; index < base::models::ResNet50::kBlockCount;
         ++index) {
        auto& block = model_.blocks[static_cast<std::size_t>(index)];
        const std::string prefix = "blocks." + std::to_string(index);
        append_conv_bn_state(result, prefix + ".conv1", block.conv1);
        append_conv_bn_state(result, prefix + ".conv2", block.conv2);
        append_conv_bn_state(result, prefix + ".conv3", block.conv3);
        if (block.has_projection) {
            append_conv_bn_state(
                result, prefix + ".projection", block.projection
            );
        }
    }
    return result;
}

std::vector<ResNet50NamedDeviceTensor>
ResNet50FeatureExtractor::stage4_parameters() {
    auto state = stage4_state_tensors();
    std::erase_if(state, [](const ResNet50NamedDeviceTensor& tensor) {
        return tensor.gradient == nullptr ||
            tensor.name.ends_with(".bias");
    });
    return state;
}

std::vector<float> ResNet50FeatureExtractor::extract(
    const base::data::image::Image& image
) {
    const __half* feature = extract_device(image);
    return download_device_feature(feature);
}

std::vector<float> ResNet50FeatureExtractor::download_device_feature(
    const __half* device_feature
) {
    return download_device_features(device_feature, 1);
}

std::vector<float> ResNet50FeatureExtractor::download_device_features(
    const __half* device_features,
    std::size_t count
) {
    if (device_features == nullptr || count == 0 || count > batch_size_) {
        throw std::invalid_argument(
            "cannot download an invalid ResNet50 feature batch"
        );
    }
    const std::size_t feature_count =
        count * static_cast<std::size_t>(kFeatureSize);
    constexpr int block_size = 256;
    half_to_float_kernel<<<
        static_cast<int>((feature_count + block_size - 1U) / block_size),
        block_size,
        0,
        stream_
    >>>(
        device_features,
        device_feature_float_,
        feature_count
    );
    check_cuda(cudaGetLastError(), "convert ResNet50 feature to FP32");

    std::vector<float> host(feature_count);
    check_cuda(
        cudaMemcpyAsync(
            host.data(),
            device_feature_float_,
            host.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_
        ),
        "download ResNet50 feature"
    );
    check_cuda(
        cudaStreamSynchronize(stream_),
        "synchronize ResNet50 feature extraction"
    );
    return host;
}

bool ResNet50FeatureExtractor::loaded() const {
    return loaded_;
}

int ResNet50FeatureExtractor::device_id() const {
    return device_id_;
}

cudaStream_t ResNet50FeatureExtractor::stream() const {
    return stream_;
}

std::size_t ResNet50FeatureExtractor::batch_size() const {
    return batch_size_;
}

}  // namespace gameagent
