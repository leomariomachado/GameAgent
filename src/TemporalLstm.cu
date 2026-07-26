#include "TemporalLstm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gameagent {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

void require_cuda_dimension(std::size_t value, const char* name) {
    if (value == 0 ||
        value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            std::string(name) + " is outside the supported CUDA range"
        );
    }
}

template <typename T>
void allocate_device(T*& destination, std::size_t count, const char* operation) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error(std::string(operation) + " size overflow");
    }
    check_cuda(
        cudaMalloc(
            reinterpret_cast<void**>(&destination),
            count * sizeof(T)
        ),
        operation
    );
}

std::vector<float> random_uniform_values(
    std::size_t count,
    float bound,
    std::mt19937& engine
) {
    std::uniform_real_distribution<float> distribution(-bound, bound);
    std::vector<float> values(count);
    for (float& value : values) {
        value = distribution(engine);
    }
    return values;
}

template <typename T>
T read_scalar(std::ifstream& input, const char* field) {
    T value{};
    input.read(
        reinterpret_cast<char*>(&value),
        static_cast<std::streamsize>(sizeof(T))
    );
    if (!input) {
        throw std::runtime_error(
            std::string("truncated temporal checkpoint field: ") + field
        );
    }
    return value;
}

std::vector<float> read_tensor(
    std::ifstream& input,
    std::size_t expected_count,
    const char* name
) {
    const std::uint64_t count =
        read_scalar<std::uint64_t>(input, name);
    if (count != expected_count) {
        throw std::runtime_error(
            std::string("temporal checkpoint tensor size mismatch: ") + name
        );
    }
    std::vector<float> values(expected_count);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
    if (!input) {
        throw std::runtime_error(
            std::string("truncated temporal checkpoint tensor: ") + name
        );
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                std::string("non-finite temporal checkpoint tensor: ") + name
            );
        }
    }
    return values;
}

__device__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__global__ void visual_projection_kernel(
    const __half* visual,
    const float* weights,
    const float* bias,
    float* projection,
    int visual_size,
    int projection_size
) {
    const int output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= projection_size) {
        return;
    }
    const float* row =
        weights + static_cast<std::size_t>(output) * visual_size;
    float sum = bias[output];
    for (int input = 0; input < visual_size; ++input) {
        sum += row[input] * __half2float(visual[input]);
    }
    projection[output] = tanhf(sum);
}

__global__ void lstm_gates_kernel(
    const float* projection,
    const float* controller,
    const float* previous_hidden,
    const float* input_weights,
    const float* recurrent_weights,
    const float* bias,
    float* gates,
    int projection_size,
    int controller_size,
    int hidden_size
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int gate_rows = hidden_size * 4;
    if (row >= gate_rows) {
        return;
    }

    const int input_size = projection_size + controller_size;
    const float* input_row =
        input_weights + static_cast<std::size_t>(row) * input_size;
    const float* recurrent_row =
        recurrent_weights + static_cast<std::size_t>(row) * hidden_size;
    float sum = bias[row];
    for (int input = 0; input < projection_size; ++input) {
        sum += input_row[input] * projection[input];
    }
    for (int input = 0; input < controller_size; ++input) {
        sum +=
            input_row[projection_size + input] * controller[input];
    }
    for (int input = 0; input < hidden_size; ++input) {
        sum += recurrent_row[input] * previous_hidden[input];
    }
    gates[row] = sum;
}

__global__ void lstm_state_kernel(
    const float* gates,
    float* hidden,
    float* cell,
    int hidden_size
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size) {
        return;
    }
    const float input_gate = sigmoid(gates[index]);
    const float forget_gate = sigmoid(gates[hidden_size + index]);
    const float candidate = tanhf(gates[2 * hidden_size + index]);
    const float output_gate = sigmoid(gates[3 * hidden_size + index]);
    const float next_cell =
        forget_gate * cell[index] + input_gate * candidate;
    cell[index] = next_cell;
    hidden[index] = output_gate * tanhf(next_cell);
}

__global__ void action_head_kernel(
    const float* hidden,
    const float* weights,
    const float* bias,
    float* logits,
    int hidden_size
) {
    const int command = blockIdx.x * blockDim.x + threadIdx.x;
    if (command >= static_cast<int>(
            TemporalLstmConfig::kControllerCommandCount
        )) {
        return;
    }
    const float* row =
        weights + static_cast<std::size_t>(command) * hidden_size;
    float value = bias[command];
    for (int input = 0; input < hidden_size; ++input) {
        value += row[input] * hidden[input];
    }
    logits[command] = value;
}

__global__ void active_head_kernel(
    const float* hidden,
    const float* weights,
    const float* bias,
    float* logit,
    int hidden_size
) {
    float value = bias[0];
    for (int input = 0; input < hidden_size; ++input) {
        value += weights[input] * hidden[input];
    }
    logit[0] = value;
}

}  // namespace

TemporalLstm::TemporalLstm(TemporalLstmConfig config)
    : config_(config),
      metadata_(config.sequence_capacity) {
    require_cuda_dimension(config_.visual_size, "visual size");
    require_cuda_dimension(
        config_.visual_projection_size, "visual projection size"
    );
    require_cuda_dimension(config_.hidden_size, "hidden size");
    require_cuda_dimension(config_.sequence_capacity, "sequence capacity");
    if (config_.visual_projection_size > 2048 ||
        config_.hidden_size > 2048 ||
        config_.sequence_capacity > 64) {
        throw std::invalid_argument(
            "temporal LSTM dimensions exceed the validated prototype limits"
        );
    }
    if (config_.device_id < 0) {
        throw std::invalid_argument("temporal LSTM device must be non-negative");
    }

    const std::size_t controller_size =
        TemporalLstmConfig::kControllerFeatureCount;
    const std::size_t input_size =
        config_.visual_projection_size + controller_size;
    const std::size_t gate_rows = config_.hidden_size * 4U;
    if (gate_rows / 4U != config_.hidden_size) {
        throw std::overflow_error("temporal LSTM gate size overflow");
    }

    try {
        check_cuda(cudaSetDevice(config_.device_id), "select temporal LSTM device");
        allocate_device(
            device_visual_ring_,
            config_.sequence_capacity * config_.visual_size,
            "allocate temporal visual ring"
        );
        allocate_device(
            device_controller_ring_,
            config_.sequence_capacity * controller_size,
            "allocate temporal controller ring"
        );
        allocate_device(
            device_projection_weights_,
            config_.visual_projection_size * config_.visual_size,
            "allocate temporal projection weights"
        );
        allocate_device(
            device_projection_bias_,
            config_.visual_projection_size,
            "allocate temporal projection bias"
        );
        allocate_device(
            device_input_weights_,
            gate_rows * input_size,
            "allocate temporal LSTM input weights"
        );
        allocate_device(
            device_recurrent_weights_,
            gate_rows * config_.hidden_size,
            "allocate temporal LSTM recurrent weights"
        );
        allocate_device(
            device_gate_bias_,
            gate_rows,
            "allocate temporal LSTM bias"
        );
        allocate_device(
            device_projection_,
            config_.visual_projection_size,
            "allocate temporal projection output"
        );
        allocate_device(
            device_gates_,
            gate_rows,
            "allocate temporal LSTM gates"
        );
        allocate_device(
            device_hidden_,
            config_.hidden_size,
            "allocate temporal LSTM hidden state"
        );
        allocate_device(
            device_cell_,
            config_.hidden_size,
            "allocate temporal LSTM cell state"
        );
        allocate_device(
            device_active_head_weights_,
            config_.hidden_size,
            "allocate temporal active-head weights"
        );
        allocate_device(
            device_active_head_bias_,
            1,
            "allocate temporal active-head bias"
        );
        allocate_device(
            device_active_logit_,
            1,
            "allocate temporal active logit"
        );
        allocate_device(
            device_action_head_weights_,
            TemporalLstmConfig::kControllerCommandCount *
                config_.hidden_size,
            "allocate temporal action-head weights"
        );
        allocate_device(
            device_action_head_bias_,
            TemporalLstmConfig::kControllerCommandCount,
            "allocate temporal action-head bias"
        );
        allocate_device(
            device_action_logits_,
            TemporalLstmConfig::kControllerCommandCount,
            "allocate temporal action logits"
        );

        std::mt19937 engine(config_.initialization_seed);
        const float projection_bound = std::sqrt(
            6.0f /
            static_cast<float>(
                config_.visual_size + config_.visual_projection_size
            )
        );
        const float input_bound = std::sqrt(
            6.0f /
            static_cast<float>(input_size + config_.hidden_size)
        );
        const float recurrent_bound = std::sqrt(
            6.0f / static_cast<float>(2U * config_.hidden_size)
        );
        const auto projection_weights = random_uniform_values(
            config_.visual_projection_size * config_.visual_size,
            projection_bound,
            engine
        );
        const auto input_weights = random_uniform_values(
            gate_rows * input_size,
            input_bound,
            engine
        );
        const auto recurrent_weights = random_uniform_values(
            gate_rows * config_.hidden_size,
            recurrent_bound,
            engine
        );
        const float head_bound = std::sqrt(
            6.0f /
            static_cast<float>(
                config_.hidden_size +
                TemporalLstmConfig::kControllerCommandCount
            )
        );
        const auto action_head_weights = random_uniform_values(
            TemporalLstmConfig::kControllerCommandCount *
                config_.hidden_size,
            head_bound,
            engine
        );
        std::array<float, TemporalLstmConfig::kControllerCommandCount>
            action_head_bias{};
        const auto active_head_weights = random_uniform_values(
            config_.hidden_size,
            std::sqrt(6.0f / static_cast<float>(config_.hidden_size + 1U)),
            engine
        );
        std::array<float, 1> active_head_bias{};
        std::vector<float> projection_bias(
            config_.visual_projection_size, 0.0f
        );
        std::vector<float> gate_bias(gate_rows, 0.0f);
        std::fill(
            gate_bias.begin() +
                static_cast<std::ptrdiff_t>(config_.hidden_size),
            gate_bias.begin() +
                static_cast<std::ptrdiff_t>(2U * config_.hidden_size),
            1.0f
        );

        check_cuda(
            cudaMemcpy(
                device_active_head_weights_,
                active_head_weights.data(),
                active_head_weights.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal active-head weights"
        );
        check_cuda(
            cudaMemcpy(
                device_active_head_bias_,
                active_head_bias.data(),
                active_head_bias.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal active-head bias"
        );
        check_cuda(
            cudaMemcpy(
                device_projection_weights_,
                projection_weights.data(),
                projection_weights.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal projection weights"
        );
        check_cuda(
            cudaMemcpy(
                device_projection_bias_,
                projection_bias.data(),
                projection_bias.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal projection bias"
        );
        check_cuda(
            cudaMemcpy(
                device_input_weights_,
                input_weights.data(),
                input_weights.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal LSTM input weights"
        );
        check_cuda(
            cudaMemcpy(
                device_recurrent_weights_,
                recurrent_weights.data(),
                recurrent_weights.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal LSTM recurrent weights"
        );
        check_cuda(
            cudaMemcpy(
                device_gate_bias_,
                gate_bias.data(),
                gate_bias.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal LSTM bias"
        );
        check_cuda(
            cudaMemcpy(
                device_action_head_weights_,
                action_head_weights.data(),
                action_head_weights.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal action-head weights"
        );
        check_cuda(
            cudaMemcpy(
                device_action_head_bias_,
                action_head_bias.data(),
                action_head_bias.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "upload temporal action-head bias"
        );
        check_cuda(
            cudaMemset(
                device_hidden_, 0, config_.hidden_size * sizeof(float)
            ),
            "initialize temporal LSTM hidden state"
        );
        check_cuda(
            cudaMemset(
                device_cell_, 0, config_.hidden_size * sizeof(float)
            ),
            "initialize temporal LSTM cell state"
        );
    } catch (...) {
        release();
        throw;
    }
}

TemporalLstm::~TemporalLstm() {
    release();
}

void TemporalLstm::release() noexcept {
    cudaSetDevice(config_.device_id);
    cudaFree(device_visual_ring_);
    cudaFree(device_controller_ring_);
    cudaFree(device_projection_weights_);
    cudaFree(device_projection_bias_);
    cudaFree(device_input_weights_);
    cudaFree(device_recurrent_weights_);
    cudaFree(device_gate_bias_);
    cudaFree(device_projection_);
    cudaFree(device_gates_);
    cudaFree(device_hidden_);
    cudaFree(device_cell_);
    cudaFree(device_active_head_weights_);
    cudaFree(device_active_head_bias_);
    cudaFree(device_active_logit_);
    cudaFree(device_action_head_weights_);
    cudaFree(device_action_head_bias_);
    cudaFree(device_action_logits_);
    device_visual_ring_ = nullptr;
    device_controller_ring_ = nullptr;
    device_projection_weights_ = nullptr;
    device_projection_bias_ = nullptr;
    device_input_weights_ = nullptr;
    device_recurrent_weights_ = nullptr;
    device_gate_bias_ = nullptr;
    device_projection_ = nullptr;
    device_gates_ = nullptr;
    device_hidden_ = nullptr;
    device_cell_ = nullptr;
    device_active_head_weights_ = nullptr;
    device_active_head_bias_ = nullptr;
    device_active_logit_ = nullptr;
    device_action_head_weights_ = nullptr;
    device_action_head_bias_ = nullptr;
    device_action_logits_ = nullptr;
    policy_has_active_head_ = false;
    policy_loaded_ = false;
}

void TemporalLstm::push(
    const __half* device_visual_feature,
    const std::array<float, 10>& applied_command_state,
    bool applied_command_known,
    float applied_command_age_seconds,
    TemporalStepMetadata metadata,
    cudaStream_t stream
) {
    if (device_visual_feature == nullptr || stream == nullptr) {
        throw std::invalid_argument(
            "temporal LSTM push requires a device feature and CUDA stream"
        );
    }
    if (metadata.frame_id == 0 || metadata.frame_timestamp_ms <= 0 ||
        metadata.decision_id == 0 ||
        metadata.decision_timestamp_ms <= 0) {
        throw std::invalid_argument(
            "temporal LSTM step has invalid frame/decision metadata"
        );
    }
    if (!std::isfinite(applied_command_age_seconds)) {
        throw std::invalid_argument(
            "temporal controller-state age must be finite"
        );
    }
    std::array<float, TemporalLstmConfig::kControllerFeatureCount>
        controller{};
    for (std::size_t index = 0;
         index < applied_command_state.size();
         ++index) {
        const float value = applied_command_state[index];
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            throw std::invalid_argument(
                "temporal controller state must be finite and in [0,1]"
            );
        }
        controller[index] = value;
    }
    controller[10] = applied_command_known ? 1.0f : 0.0f;
    controller[11] = std::clamp(
        applied_command_age_seconds, 0.0f, 1.0f
    );
    metadata.applied_command_known = applied_command_known;

    check_cuda(cudaSetDevice(config_.device_id), "select temporal push device");
    __half* visual_destination =
        device_visual_ring_ + next_slot_ * config_.visual_size;
    float* controller_destination =
        device_controller_ring_ +
        next_slot_ * TemporalLstmConfig::kControllerFeatureCount;
    check_cuda(
        cudaMemcpyAsync(
            visual_destination,
            device_visual_feature,
            config_.visual_size * sizeof(__half),
            cudaMemcpyDeviceToDevice,
            stream
        ),
        "copy ResNet50 feature into temporal ring"
    );
    check_cuda(
        cudaMemcpyAsync(
            controller_destination,
            controller.data(),
            controller.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        ),
        "copy confirmed controller state into temporal ring"
    );

    metadata_[next_slot_] = metadata;
    next_slot_ = (next_slot_ + 1U) % config_.sequence_capacity;
    size_ = std::min(size_ + 1U, config_.sequence_capacity);
}

void TemporalLstm::launch_forward(cudaStream_t stream) {
    check_cuda(
        cudaMemsetAsync(
            device_hidden_, 0, config_.hidden_size * sizeof(float), stream
        ),
        "reset temporal LSTM hidden state"
    );
    check_cuda(
        cudaMemsetAsync(
            device_cell_, 0, config_.hidden_size * sizeof(float), stream
        ),
        "reset temporal LSTM cell state"
    );

    constexpr int block_size = 128;
    const int visual_size = static_cast<int>(config_.visual_size);
    const int projection_size =
        static_cast<int>(config_.visual_projection_size);
    const int hidden_size = static_cast<int>(config_.hidden_size);
    const int controller_size =
        static_cast<int>(TemporalLstmConfig::kControllerFeatureCount);
    const std::size_t first_slot =
        size_ == config_.sequence_capacity ? next_slot_ : 0U;
    for (std::size_t step = 0; step < size_; ++step) {
        const std::size_t slot =
            (first_slot + step) % config_.sequence_capacity;
        const __half* visual =
            device_visual_ring_ + slot * config_.visual_size;
        const float* controller =
            device_controller_ring_ +
            slot * TemporalLstmConfig::kControllerFeatureCount;

        visual_projection_kernel<<<
            (projection_size + block_size - 1) / block_size,
            block_size,
            0,
            stream
        >>>(
            visual,
            device_projection_weights_,
            device_projection_bias_,
            device_projection_,
            visual_size,
            projection_size
        );
        check_cuda(cudaGetLastError(), "launch temporal visual projection");

        const int gate_rows = hidden_size * 4;
        lstm_gates_kernel<<<
            (gate_rows + block_size - 1) / block_size,
            block_size,
            0,
            stream
        >>>(
            device_projection_,
            controller,
            device_hidden_,
            device_input_weights_,
            device_recurrent_weights_,
            device_gate_bias_,
            device_gates_,
            projection_size,
            controller_size,
            hidden_size
        );
        check_cuda(cudaGetLastError(), "launch temporal LSTM gates");

        lstm_state_kernel<<<
            (hidden_size + block_size - 1) / block_size,
            block_size,
            0,
            stream
        >>>(
            device_gates_,
            device_hidden_,
            device_cell_,
            hidden_size
        );
        check_cuda(cudaGetLastError(), "launch temporal LSTM state update");
    }
}

TemporalLstmOutput TemporalLstm::infer(cudaStream_t stream) {
    const auto started = std::chrono::steady_clock::now();
    const float* device_output = infer_device(stream);
    TemporalLstmOutput result;
    result.hidden.resize(config_.hidden_size);
    check_cuda(
        cudaMemcpyAsync(
            result.hidden.data(),
            device_output,
            result.hidden.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        ),
        "download temporal LSTM hidden state"
    );
    check_cuda(
        cudaStreamSynchronize(stream),
        "synchronize temporal LSTM inference"
    );
    const auto finished = std::chrono::steady_clock::now();
    result.inference_ms =
        std::chrono::duration<double, std::milli>(
            finished - started
        ).count();

    const std::size_t first_slot =
        size_ == config_.sequence_capacity ? next_slot_ : 0U;
    result.steps.reserve(size_);
    for (std::size_t step = 0; step < size_; ++step) {
        result.steps.push_back(
            metadata_[(first_slot + step) % config_.sequence_capacity]
        );
    }
    for (float value : result.hidden) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "temporal LSTM produced a non-finite hidden state"
            );
        }
    }
    return result;
}

const float* TemporalLstm::infer_device(cudaStream_t stream) {
    if (!ready()) {
        throw std::runtime_error(
            "temporal LSTM inference requires at least one observation"
        );
    }
    if (stream == nullptr) {
        throw std::invalid_argument(
            "temporal LSTM inference requires a CUDA stream"
        );
    }
    check_cuda(
        cudaSetDevice(config_.device_id), "select temporal inference device"
    );
    launch_forward(stream);
    return device_hidden_;
}

TemporalPolicyCheckpointReport TemporalLstm::load_checkpoint(
    const std::string& path
) {
    if (path.empty()) {
        throw std::invalid_argument(
            "temporal policy checkpoint path is empty"
        );
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "could not open temporal policy checkpoint: " + path
        );
    }
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    constexpr std::array<char, 8> expected_magic{
        'G', 'A', 'T', 'L', 'S', 'T', 'M', '1'
    };
    if (!input || magic != expected_magic) {
        throw std::runtime_error(
            "invalid temporal policy checkpoint magic"
        );
    }
    const std::uint32_t version =
        read_scalar<std::uint32_t>(input, "version");
    const std::uint32_t visual_size =
        read_scalar<std::uint32_t>(input, "visual_size");
    const std::uint32_t projection_size =
        read_scalar<std::uint32_t>(input, "projection_size");
    const std::uint32_t controller_size =
        read_scalar<std::uint32_t>(input, "controller_size");
    const std::uint32_t hidden_size =
        read_scalar<std::uint32_t>(input, "hidden_size");
    const std::uint32_t output_size =
        read_scalar<std::uint32_t>(input, "output_size");
    const std::uint64_t training_step =
        read_scalar<std::uint64_t>(input, "training_step");
    const std::uint64_t epoch =
        read_scalar<std::uint64_t>(input, "epoch");
    const float best_validation_loss =
        read_scalar<float>(input, "best_validation_loss");
    float best_selection_score = 0.0f;
    float active_threshold = 0.5f;
    float command_threshold = 0.6f;
    if (version == 2) {
        best_selection_score =
            read_scalar<float>(input, "best_selection_score");
        active_threshold =
            read_scalar<float>(input, "active_threshold");
        command_threshold =
            read_scalar<float>(input, "command_threshold");
    }
    if ((version != 1 && version != 2) ||
        visual_size != config_.visual_size ||
        projection_size != config_.visual_projection_size ||
        controller_size !=
            TemporalLstmConfig::kControllerFeatureCount ||
        hidden_size != config_.hidden_size ||
        output_size !=
            TemporalLstmConfig::kControllerCommandCount ||
        !std::isfinite(best_validation_loss) ||
        !std::isfinite(best_selection_score) ||
        !std::isfinite(active_threshold) ||
        !std::isfinite(command_threshold) ||
        active_threshold <= 0.0f || active_threshold >= 1.0f ||
        command_threshold <= 0.0f || command_threshold >= 1.0f) {
        throw std::runtime_error(
            "temporal policy checkpoint architecture mismatch"
        );
    }

    const std::size_t gate_rows = config_.hidden_size * 4U;
    const std::size_t recurrent_input_size =
        config_.visual_projection_size +
        TemporalLstmConfig::kControllerFeatureCount;
    const auto projection_weights = read_tensor(
        input,
        config_.visual_projection_size * config_.visual_size,
        "visual_projection.weight"
    );
    const auto projection_bias = read_tensor(
        input,
        config_.visual_projection_size,
        "visual_projection.bias"
    );
    const auto input_weights = read_tensor(
        input,
        gate_rows * recurrent_input_size,
        "lstm.weight_ih"
    );
    const auto recurrent_weights = read_tensor(
        input,
        gate_rows * config_.hidden_size,
        "lstm.weight_hh"
    );
    const auto gate_bias = read_tensor(
        input, gate_rows, "lstm.bias"
    );
    std::vector<float> active_head_weights;
    std::vector<float> active_head_bias;
    if (version == 2) {
        active_head_weights = read_tensor(
            input, config_.hidden_size, "active_head.weight"
        );
        active_head_bias = read_tensor(
            input, 1, "active_head.bias"
        );
    }
    const auto action_head_weights = read_tensor(
        input,
        TemporalLstmConfig::kControllerCommandCount *
            config_.hidden_size,
        "action_head.weight"
    );
    const auto action_head_bias = read_tensor(
        input,
        TemporalLstmConfig::kControllerCommandCount,
        "action_head.bias"
    );
    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error(
            "temporal policy checkpoint has unexpected trailing data"
        );
    }

    check_cuda(
        cudaSetDevice(config_.device_id),
        "select temporal checkpoint device"
    );
    const auto upload = [](
        float* destination,
        const std::vector<float>& source,
        const char* operation
    ) {
        check_cuda(
            cudaMemcpy(
                destination,
                source.data(),
                source.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            operation
        );
    };
    upload(
        device_projection_weights_, projection_weights,
        "upload trained temporal projection weights"
    );
    upload(
        device_projection_bias_, projection_bias,
        "upload trained temporal projection bias"
    );
    upload(
        device_input_weights_, input_weights,
        "upload trained temporal input weights"
    );
    upload(
        device_recurrent_weights_, recurrent_weights,
        "upload trained temporal recurrent weights"
    );
    upload(
        device_gate_bias_, gate_bias,
        "upload trained temporal gate bias"
    );
    if (version == 2) {
        upload(
            device_active_head_weights_, active_head_weights,
            "upload trained temporal active-head weights"
        );
        upload(
            device_active_head_bias_, active_head_bias,
            "upload trained temporal active-head bias"
        );
    }
    upload(
        device_action_head_weights_, action_head_weights,
        "upload trained temporal action-head weights"
    );
    upload(
        device_action_head_bias_, action_head_bias,
        "upload trained temporal action-head bias"
    );
    active_threshold_ = active_threshold;
    command_threshold_ = command_threshold;
    policy_has_active_head_ = version == 2;
    policy_loaded_ = true;
    return {
        .version = version,
        .training_step = training_step,
        .epoch = epoch,
        .best_validation_loss = best_validation_loss,
        .best_selection_score = best_selection_score,
        .active_threshold = active_threshold_,
        .command_threshold = command_threshold_,
        .parameter_count =
            projection_weights.size() +
            projection_bias.size() +
            input_weights.size() +
            recurrent_weights.size() +
            gate_bias.size() +
            active_head_weights.size() +
            active_head_bias.size() +
            action_head_weights.size() +
            action_head_bias.size(),
    };
}

TemporalActionPrediction TemporalLstm::predict_from_hidden(
    cudaStream_t stream,
    float threshold_override
) {
    if (!policy_loaded_) {
        throw std::runtime_error(
            "temporal action prediction requires a trained checkpoint"
        );
    }
    if (stream == nullptr || !std::isfinite(threshold_override) ||
        (threshold_override > 0.0f && threshold_override >= 1.0f)) {
        throw std::invalid_argument(
            "invalid temporal action prediction configuration"
        );
    }
    if (policy_has_active_head_) {
        active_head_kernel<<<1, 1, 0, stream>>>(
            device_hidden_,
            device_active_head_weights_,
            device_active_head_bias_,
            device_active_logit_,
            static_cast<int>(config_.hidden_size)
        );
        check_cuda(cudaGetLastError(), "launch temporal active head");
    }
    action_head_kernel<<<1, 32, 0, stream>>>(
        device_hidden_,
        device_action_head_weights_,
        device_action_head_bias_,
        device_action_logits_,
        static_cast<int>(config_.hidden_size)
    );
    check_cuda(cudaGetLastError(), "launch temporal action head");
    std::array<float, TemporalLstmConfig::kControllerCommandCount>
        logits{};
    float active_logit = 0.0f;
    if (policy_has_active_head_) {
        check_cuda(
            cudaMemcpyAsync(
                &active_logit,
                device_active_logit_,
                sizeof(float),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "download temporal active logit"
        );
    }
    check_cuda(
        cudaMemcpyAsync(
            logits.data(),
            device_action_logits_,
            logits.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        ),
        "download temporal action logits"
    );
    check_cuda(
        cudaStreamSynchronize(stream),
        "synchronize temporal action prediction"
    );
    TemporalActionPrediction prediction;
    prediction.active_probability = policy_has_active_head_
        ? 1.0f / (1.0f + std::exp(-active_logit))
        : 1.0f;
    prediction.action_active =
        prediction.active_probability >= active_threshold_;
    const float command_threshold =
        threshold_override > 0.0f
            ? threshold_override
            : command_threshold_;
    std::vector<std::pair<float, int>> candidates;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        const float probability =
            1.0f / (1.0f + std::exp(-logits[index]));
        prediction.probabilities[index] = probability;
        if (prediction.action_active &&
            probability >= command_threshold) {
            candidates.emplace_back(
                probability, static_cast<int>(index)
            );
        }
    }
    std::sort(
        candidates.begin(), candidates.end(),
        [](const auto& left, const auto& right) {
            return left.first > right.first;
        }
    );
    if (prediction.action_active && candidates.empty()) {
        const auto strongest = std::max_element(
            prediction.probabilities.begin(),
            prediction.probabilities.end()
        );
        candidates.emplace_back(
            *strongest,
            static_cast<int>(
                std::distance(
                    prediction.probabilities.begin(), strongest
                )
            )
        );
    }
    const auto conflicts = [](int left, int right) {
        return
            (left == 6 && right == 7) ||
            (left == 7 && right == 6) ||
            (left == 8 && right == 9) ||
            (left == 9 && right == 8);
    };
    for (const auto& [probability, command] : candidates) {
        static_cast<void>(probability);
        const bool contradictory = std::any_of(
            prediction.commands.begin(),
            prediction.commands.end(),
            [&](int selected) { return conflicts(selected, command); }
        );
        if (!contradictory) {
            prediction.commands.push_back(command);
        }
        if (prediction.commands.size() == 2) {
            break;
        }
    }
    std::sort(
        prediction.commands.begin(), prediction.commands.end()
    );
    return prediction;
}

void TemporalLstm::reset(cudaStream_t stream) {
    next_slot_ = 0;
    size_ = 0;
    std::fill(metadata_.begin(), metadata_.end(), TemporalStepMetadata{});
    if (stream != nullptr) {
        check_cuda(
            cudaSetDevice(config_.device_id), "select temporal reset device"
        );
        check_cuda(
            cudaMemsetAsync(
                device_hidden_, 0, config_.hidden_size * sizeof(float), stream
            ),
            "clear temporal hidden state"
        );
        check_cuda(
            cudaMemsetAsync(
                device_cell_, 0, config_.hidden_size * sizeof(float), stream
            ),
            "clear temporal cell state"
        );
    }
}

bool TemporalLstm::ready() const {
    return size_ != 0;
}

bool TemporalLstm::policy_loaded() const {
    return policy_loaded_;
}

std::size_t TemporalLstm::size() const {
    return size_;
}

std::size_t TemporalLstm::capacity() const {
    return config_.sequence_capacity;
}

std::size_t TemporalLstm::hidden_size() const {
    return config_.hidden_size;
}

const TemporalLstmConfig& TemporalLstm::config() const {
    return config_;
}

}  // namespace gameagent
