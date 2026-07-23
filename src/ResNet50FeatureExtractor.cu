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

struct NamedDeviceTensor {
    std::string name;
    float* data = nullptr;
    std::size_t size = 0;
};

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
    std::vector<NamedDeviceTensor>& destination,
    const std::string& prefix,
    base::models::blocks::ConvBnRelu& layer
) {
    const std::size_t channels =
        static_cast<std::size_t>(layer.bn_params.channels);
    destination.push_back({
        prefix + ".weight",
        layer.weight,
        convolution_weight_size(layer.conv_params),
    });
    destination.push_back({
        prefix + ".bias",
        layer.bias,
        static_cast<std::size_t>(layer.conv_params.out_channels),
    });
    destination.push_back({prefix + ".gamma", layer.gamma, channels});
    destination.push_back({prefix + ".beta", layer.beta, channels});
    destination.push_back({
        prefix + ".running_mean", layer.running_mean, channels,
    });
    destination.push_back({
        prefix + ".running_var", layer.running_var, channels,
    });
}

std::vector<NamedDeviceTensor> model_state_tensors(
    base::models::ResNet50& model
) {
    std::vector<NamedDeviceTensor> tensors;
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
        model.classifier.weight_size(),
    });
    tensors.push_back({
        "classifier.bias",
        model.classifier.bias,
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
    bool verify_checkpoint_copy
)
    : device_id_(device_id),
      verify_checkpoint_copy_(verify_checkpoint_copy) {
    if (device_id_ < 0) {
        throw std::invalid_argument("ResNet50 inference device must be non-negative");
    }

    try {
        check_cuda(cudaSetDevice(device_id_), "set ResNet50 inference device");
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "create ResNet50 inference stream"
        );
        model_.allocate(
            1,
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
        host_input_.resize(input_count);
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_input_float_),
                input_count * sizeof(float)
            ),
            "allocate ResNet50 FP32 input"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_input_half_),
                input_count * sizeof(__half)
            ),
            "allocate ResNet50 FP16 input"
        );
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&device_feature_float_),
                static_cast<std::size_t>(kFeatureSize) * sizeof(float)
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
    const base::data::image::Image& image
) {
    if (image.empty() || image.channels() != kInputChannels) {
        throw std::invalid_argument(
            "ResNet50 inference requires a non-empty RGB image"
        );
    }

    if (image.width() == kInputWidth && image.height() == kInputHeight) {
        base::data::image::image_to_chw_float(
            image, host_input_.data(), host_input_.size()
        );
    } else {
        const auto resized = base::data::image::resize_bilinear(
            image, kInputWidth, kInputHeight
        );
        base::data::image::image_to_chw_float(
            resized, host_input_.data(), host_input_.size()
        );
    }
    normalize_imagenet_chw(host_input_);
}

const __half* ResNet50FeatureExtractor::extract_device(
    const base::data::image::Image& image
) {
    if (!loaded_) {
        throw std::runtime_error(
            "ResNet50 features requested before loading a checkpoint"
        );
    }

    prepare_input(image);
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

std::vector<float> ResNet50FeatureExtractor::extract(
    const base::data::image::Image& image
) {
    const __half* feature = extract_device(image);
    constexpr int block_size = 256;
    half_to_float_kernel<<<
        (kFeatureSize + block_size - 1) / block_size,
        block_size,
        0,
        stream_
    >>>(
        feature,
        device_feature_float_,
        static_cast<std::size_t>(kFeatureSize)
    );
    check_cuda(cudaGetLastError(), "convert ResNet50 feature to FP32");

    std::vector<float> host(static_cast<std::size_t>(kFeatureSize));
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

}  // namespace gameagent
