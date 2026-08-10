#include "CppRecurrentPolicyBackend.h"
#include "ResNet50FeatureExtractor.h"

#include "checkpoint/checkpoint.h"
#include "optim/optimizer/adamw.h"
#include "stream.h"
#include "tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gameagent {
namespace {

constexpr float kAdamWeightDecay = 1.0e-5f;
constexpr float kNegativeInfinity = -1.0e30f;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}

const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "success";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
        case CUBLAS_STATUS_MAPPING_ERROR: return "mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
        default: return "unknown";
    }
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + ": " + cublas_status_name(status)
        );
    }
}

void require_positive(std::size_t value, const char* name) {
    if (value == 0) {
        throw std::invalid_argument(
            std::string("native recurrent ") + name + " must be positive"
        );
    }
}

float* float_data(base::Tensor& tensor) {
    return static_cast<float*>(tensor.data());
}

const float* float_data(const base::Tensor& tensor) {
    return static_cast<const float*>(tensor.data());
}

int* int_data(base::Tensor& tensor) {
    return static_cast<int*>(tensor.data());
}

const int* int_data(const base::Tensor& tensor) {
    return static_cast<const int*>(tensor.data());
}

struct DeviceParameter {
    std::string name;
    std::size_t size = 0;
    bool trainable = false;
    base::Tensor value;
    base::Tensor gradient;
    base::Tensor first_moment;
    base::Tensor second_moment;

    void allocate(std::string value_name, std::size_t value_size,
                  bool value_trainable, int device_id) {
        name = std::move(value_name);
        size = value_size;
        trainable = value_trainable;
        value = base::Tensor::zeros({size}, base::DType::F32, device_id);
        if (trainable) {
            gradient = base::Tensor::zeros(
                {size}, base::DType::F32, device_id
            );
            first_moment = base::Tensor::zeros(
                {size}, base::DType::F32, device_id
            );
            second_moment = base::Tensor::zeros(
                {size}, base::DType::F32, device_id
            );
        }
    }

    float* values() { return float_data(value); }
    const float* values() const { return float_data(value); }
    float* gradients() {
        return trainable ? float_data(gradient) : nullptr;
    }
    float* moments1() {
        return trainable ? float_data(first_moment) : nullptr;
    }
    float* moments2() {
        return trainable ? float_data(second_moment) : nullptr;
    }
};

struct Network {
    DeviceParameter input_weight;
    DeviceParameter recurrent_weight;
    DeviceParameter input_bias;
    DeviceParameter recurrent_bias;
    DeviceParameter character_embedding;
    DeviceParameter opponent_embedding;
    DeviceParameter strategy_weight;
    DeviceParameter strategy_bias;
    DeviceParameter head_weight;
    DeviceParameter head_bias;

    std::vector<DeviceParameter*> parameters() {
        return {
            &input_weight,
            &recurrent_weight,
            &input_bias,
            &recurrent_bias,
            &character_embedding,
            &opponent_embedding,
            &strategy_weight,
            &strategy_bias,
            &head_weight,
            &head_bias,
        };
    }

    std::vector<const DeviceParameter*> parameters() const {
        return {
            &input_weight,
            &recurrent_weight,
            &input_bias,
            &recurrent_bias,
            &character_embedding,
            &opponent_embedding,
            &strategy_weight,
            &strategy_bias,
            &head_weight,
            &head_bias,
        };
    }
};

struct SequenceSnapshot {
    std::size_t length = 0;
    std::size_t input_size = 0;
    std::vector<__half> values;
};

struct EncodedVisualFrame {
    std::uint64_t frame_id = 0;
    std::vector<std::uint8_t> jpeg;
};

struct Transition {
    std::shared_ptr<const SequenceSnapshot> state_sequence;
    std::vector<float> state_context;
    std::shared_ptr<const SequenceSnapshot> next_sequence;
    std::vector<float> next_context;
    std::vector<std::uint8_t> next_action_mask;
    std::size_t action = 0;
    std::size_t duration = 1;
    float reward = 0.0f;
    bool terminal = false;
    bool successful_episode = false;
    std::size_t character = 0;
    std::size_t opponent = 0;
    float selected_q = 0.0f;
    std::shared_ptr<EncodedVisualFrame> state_jpeg;
};

struct ActiveTransition {
    std::shared_ptr<const SequenceSnapshot> sequence;
    std::vector<float> context;
    std::size_t action = 0;
    std::size_t character = 0;
    std::size_t opponent = 0;
    float selected_q = 0.0f;
    std::shared_ptr<EncodedVisualFrame> state_jpeg;
};

struct BatchHost {
    std::size_t batch = 0;
    std::size_t steps = 0;
    std::size_t input = 0;
    std::vector<float> sequence;
    std::vector<int> lengths;
    std::vector<float> context;
    std::vector<int> characters;
    std::vector<int> opponents;
};

struct ForwardWorkspace {
    std::size_t batch_capacity = 0;
    std::size_t step_capacity = 0;
    std::size_t input_size = 0;
    std::size_t hidden_size = 0;
    std::size_t context_size = 0;
    std::size_t joined_size = 0;
    std::size_t strategy_size = 0;
    std::size_t max_actions = 0;
    base::Tensor sequence;
    base::Tensor lengths;
    base::Tensor context;
    base::Tensor characters;
    base::Tensor opponents;
    base::Tensor hidden;
    base::Tensor cell;
    base::Tensor gates;
    base::Tensor joined;
    base::Tensor strategy_pre;
    base::Tensor strategy;
    base::Tensor q_values;
    base::Tensor d_strategy_pre;
    base::Tensor d_joined;
    base::Tensor d_hidden;
    base::Tensor d_cell;
    base::Tensor d_gates;
    base::Tensor d_hidden_previous;
    base::Tensor d_cell_previous;
    base::Tensor gradient_q;
    base::Tensor d_sequence;

    void allocate(
        const RecurrentPolicyBackendConfig& config,
        std::size_t capacity,
        int device_id
    ) {
        batch_capacity = capacity;
        step_capacity = config.sequence_capacity;
        input_size = config.projection_size + config.controller_size;
        hidden_size = config.temporal_hidden_size;
        context_size = config.context_size;
        joined_size = hidden_size + context_size +
            config.character_embedding_size +
            config.opponent_embedding_size;
        strategy_size = config.strategy_hidden_size;
        max_actions = *std::max_element(
            config.action_sizes.begin(), config.action_sizes.end()
        );
        sequence = base::Tensor::zeros(
            {step_capacity * batch_capacity * input_size},
            base::DType::F32, device_id
        );
        lengths = base::Tensor::zeros(
            {batch_capacity}, base::DType::I32, device_id
        );
        context = base::Tensor::zeros(
            {batch_capacity * context_size}, base::DType::F32, device_id
        );
        characters = base::Tensor::zeros(
            {batch_capacity}, base::DType::I32, device_id
        );
        opponents = base::Tensor::zeros(
            {batch_capacity}, base::DType::I32, device_id
        );
        hidden = base::Tensor::zeros(
            {(step_capacity + 1) * batch_capacity * hidden_size},
            base::DType::F32, device_id
        );
        cell = base::Tensor::zeros(
            {(step_capacity + 1) * batch_capacity * hidden_size},
            base::DType::F32, device_id
        );
        gates = base::Tensor::zeros(
            {step_capacity * batch_capacity * 4 * hidden_size},
            base::DType::F32, device_id
        );
        joined = base::Tensor::zeros(
            {batch_capacity * joined_size}, base::DType::F32, device_id
        );
        strategy_pre = base::Tensor::zeros(
            {batch_capacity * strategy_size}, base::DType::F32, device_id
        );
        strategy = base::Tensor::zeros(
            {batch_capacity * strategy_size}, base::DType::F32, device_id
        );
        q_values = base::Tensor::zeros(
            {batch_capacity * max_actions}, base::DType::F32, device_id
        );
        d_strategy_pre = base::Tensor::zeros(
            {batch_capacity * strategy_size}, base::DType::F32, device_id
        );
        d_joined = base::Tensor::zeros(
            {batch_capacity * joined_size}, base::DType::F32, device_id
        );
        d_hidden = base::Tensor::zeros(
            {batch_capacity * hidden_size}, base::DType::F32, device_id
        );
        d_cell = base::Tensor::zeros(
            {batch_capacity * hidden_size}, base::DType::F32, device_id
        );
        d_gates = base::Tensor::zeros(
            {batch_capacity * 4 * hidden_size}, base::DType::F32, device_id
        );
        d_hidden_previous = base::Tensor::zeros(
            {batch_capacity * hidden_size}, base::DType::F32, device_id
        );
        d_cell_previous = base::Tensor::zeros(
            {batch_capacity * hidden_size}, base::DType::F32, device_id
        );
        gradient_q = base::Tensor::zeros(
            {batch_capacity}, base::DType::F32, device_id
        );
        d_sequence = base::Tensor::zeros(
            {step_capacity * batch_capacity * input_size},
            base::DType::F32, device_id
        );
    }
};

__device__ float sigmoid_value(float value) {
    return 1.0f / (1.0f + expf(-value));
}

__global__ void projection_kernel(
    const float* visual,
    const float* weights,
    const float* bias,
    float* output,
    int visual_size,
    int projection_size
) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= projection_size) {
        return;
    }
    float sum = bias[row];
    const float* row_weight = weights +
        static_cast<std::size_t>(row) * visual_size;
    for (int column = 0; column < visual_size; ++column) {
        sum += row_weight[column] * visual[column];
    }
    output[row] = tanhf(sum);
}

__global__ void projection_half_batch_kernel(
    const __half* visual,
    const float* weights,
    const float* bias,
    float* output,
    int batch,
    int visual_size,
    int projection_size
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= batch * projection_size) {
        return;
    }
    const int row = index / projection_size;
    const int output_index = index - row * projection_size;
    float sum = bias[output_index];
    const float* row_weight = weights +
        static_cast<std::size_t>(output_index) * visual_size;
    const __half* row_visual = visual +
        static_cast<std::size_t>(row) * visual_size;
    for (int column = 0; column < visual_size; ++column) {
        sum += row_weight[column] * __half2float(row_visual[column]);
    }
    output[index] = tanhf(sum);
}

__global__ void projection_feature_gradient_kernel(
    const float* projected_gradient,
    const float* projected,
    const float* projection_weight,
    float* feature_gradient,
    int batch,
    int projection_size,
    int visual_size
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= batch * visual_size) {
        return;
    }
    const int row = index / visual_size;
    const int feature = index - row * visual_size;
    float sum = 0.0f;
    for (int output = 0; output < projection_size; ++output) {
        const float activation = projected[
            static_cast<std::size_t>(row) * projection_size + output
        ];
        sum += projected_gradient[
            static_cast<std::size_t>(row) * projection_size + output
        ] * (1.0f - activation * activation) * projection_weight[
            static_cast<std::size_t>(output) * visual_size + feature
        ];
    }
    feature_gradient[index] = sum;
}

__global__ void lstm_state_kernel(
    float* gates,
    const float* input_bias,
    const float* recurrent_bias,
    const float* hidden_previous,
    const float* cell_previous,
    float* hidden_next,
    float* cell_next,
    const int* lengths,
    int timestep,
    int batch,
    int hidden
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch * hidden;
    if (index >= total) {
        return;
    }
    const int row = index / hidden;
    const int unit = index - row * hidden;
    if (timestep >= lengths[row]) {
        hidden_next[index] = hidden_previous[index];
        cell_next[index] = cell_previous[index];
        gates[static_cast<std::size_t>(row) * 4 * hidden + unit] = 0.0f;
        gates[static_cast<std::size_t>(row) * 4 * hidden + hidden + unit] = 0.0f;
        gates[static_cast<std::size_t>(row) * 4 * hidden + 2 * hidden + unit] = 0.0f;
        gates[static_cast<std::size_t>(row) * 4 * hidden + 3 * hidden + unit] = 0.0f;
        return;
    }
    const std::size_t base =
        static_cast<std::size_t>(row) * 4 * hidden + unit;
    const float input_gate = sigmoid_value(
        gates[base] + input_bias[unit] + recurrent_bias[unit]
    );
    const float forget_gate = sigmoid_value(
        gates[base + hidden] + input_bias[hidden + unit] +
        recurrent_bias[hidden + unit]
    );
    const float candidate = tanhf(
        gates[base + 2 * hidden] + input_bias[2 * hidden + unit] +
        recurrent_bias[2 * hidden + unit]
    );
    const float output_gate = sigmoid_value(
        gates[base + 3 * hidden] + input_bias[3 * hidden + unit] +
        recurrent_bias[3 * hidden + unit]
    );
    gates[base] = input_gate;
    gates[base + hidden] = forget_gate;
    gates[base + 2 * hidden] = candidate;
    gates[base + 3 * hidden] = output_gate;
    const float next_cell =
        forget_gate * cell_previous[index] + input_gate * candidate;
    cell_next[index] = next_cell;
    hidden_next[index] = output_gate * tanhf(next_cell);
}

__global__ void joined_kernel(
    const float* hidden_states,
    const int* lengths,
    const float* context,
    const int* characters,
    const int* opponents,
    const float* character_embedding,
    const float* opponent_embedding,
    float* joined,
    int batch,
    int hidden,
    int context_size,
    int character_embedding_size,
    int opponent_embedding_size,
    int joined_size
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = batch * joined_size;
    if (index >= total) {
        return;
    }
    const int row = index / joined_size;
    const int column = index - row * joined_size;
    if (column < hidden) {
        const int step = lengths[row];
        joined[index] = hidden_states[
            (static_cast<std::size_t>(step) * batch + row) * hidden +
            column
        ];
        return;
    }
    int offset = column - hidden;
    if (offset < context_size) {
        joined[index] = context[
            static_cast<std::size_t>(row) * context_size + offset
        ];
        return;
    }
    offset -= context_size;
    if (offset < character_embedding_size) {
        joined[index] = character_embedding[
            static_cast<std::size_t>(characters[row]) *
                character_embedding_size + offset
        ];
        return;
    }
    offset -= character_embedding_size;
    joined[index] = opponent_embedding[
        static_cast<std::size_t>(opponents[row]) *
            opponent_embedding_size + offset
    ];
}

__global__ void bias_relu_kernel(
    float* input,
    const float* bias,
    float* output,
    int batch,
    int width
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= batch * width) {
        return;
    }
    const float value = input[index] + bias[index % width];
    input[index] = value;
    output[index] = value > 0.0f ? value : 0.0f;
}

__global__ void q_head_kernel(
    const float* strategy,
    const int* characters,
    const int* action_sizes,
    const float* head_weight,
    const float* head_bias,
    float* q_values,
    int batch,
    int strategy_size,
    int max_actions
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= batch * max_actions) {
        return;
    }
    const int row = index / max_actions;
    const int action = index - row * max_actions;
    const int character = characters[row];
    if (action >= action_sizes[character]) {
        q_values[index] = kNegativeInfinity;
        return;
    }
    const std::size_t head =
        (static_cast<std::size_t>(character) * max_actions + action) *
        strategy_size;
    float value = head_bias[
        static_cast<std::size_t>(character) * max_actions + action
    ];
    const float* latent = strategy +
        static_cast<std::size_t>(row) * strategy_size;
    for (int column = 0; column < strategy_size; ++column) {
        value += head_weight[head + column] * latent[column];
    }
    q_values[index] = value;
}

__global__ void head_backward_kernel(
    const float* strategy,
    const float* strategy_pre,
    const int* characters,
    const int* actions,
    const float* gradient_q,
    const float* head_weight,
    float* head_weight_gradient,
    float* head_bias_gradient,
    float* strategy_gradient,
    int batch,
    int strategy_size,
    int max_actions
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= batch * strategy_size) {
        return;
    }
    const int row = index / strategy_size;
    const int column = index - row * strategy_size;
    const int character = characters[row];
    const int action = actions[row];
    const std::size_t head =
        (static_cast<std::size_t>(character) * max_actions + action) *
        strategy_size;
    const float grad = gradient_q[row];
    atomicAdd(
        head_weight_gradient + head + column,
        grad * strategy[index]
    );
    if (column == 0) {
        atomicAdd(
            head_bias_gradient +
                static_cast<std::size_t>(character) * max_actions + action,
            grad
        );
    }
    const float latent_gradient = grad * head_weight[head + column];
    strategy_gradient[index] =
        strategy_pre[index] > 0.0f ? latent_gradient : 0.0f;
}

__global__ void strategy_weight_gradient_kernel(
    const float* strategy_gradient,
    const float* joined,
    float* weight_gradient,
    int batch,
    int strategy_size,
    int joined_size
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= strategy_size * joined_size) {
        return;
    }
    const int output = index / joined_size;
    const int input = index - output * joined_size;
    float sum = 0.0f;
    for (int row = 0; row < batch; ++row) {
        sum += strategy_gradient[
            static_cast<std::size_t>(row) * strategy_size + output
        ] * joined[static_cast<std::size_t>(row) * joined_size + input];
    }
    weight_gradient[index] += sum;
}

__global__ void strategy_bias_gradient_kernel(
    const float* strategy_gradient,
    float* bias_gradient,
    int batch,
    int strategy_size
) {
    const int output = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (output >= strategy_size) {
        return;
    }
    float sum = 0.0f;
    for (int row = 0; row < batch; ++row) {
        sum += strategy_gradient[
            static_cast<std::size_t>(row) * strategy_size + output
        ];
    }
    bias_gradient[output] += sum;
}

__global__ void embedding_and_hidden_gradient_kernel(
    const float* joined_gradient,
    const int* characters,
    const int* opponents,
    float* character_gradient,
    float* opponent_gradient,
    float* hidden_gradient,
    int batch,
    int hidden,
    int context_size,
    int character_embedding_size,
    int opponent_embedding_size,
    int joined_size
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int feature_count = hidden + character_embedding_size +
        opponent_embedding_size;
    if (index >= batch * feature_count) {
        return;
    }
    const int row = index / feature_count;
    int column = index - row * feature_count;
    if (column < hidden) {
        hidden_gradient[
            static_cast<std::size_t>(row) * hidden + column
        ] = joined_gradient[
            static_cast<std::size_t>(row) * joined_size + column
        ];
        return;
    }
    column -= hidden;
    if (column < character_embedding_size) {
        atomicAdd(
            character_gradient +
                static_cast<std::size_t>(characters[row]) *
                    character_embedding_size + column,
            joined_gradient[
                static_cast<std::size_t>(row) * joined_size + hidden +
                context_size + column
            ]
        );
        return;
    }
    column -= character_embedding_size;
    atomicAdd(
        opponent_gradient +
            static_cast<std::size_t>(opponents[row]) *
                opponent_embedding_size + column,
        joined_gradient[
            static_cast<std::size_t>(row) * joined_size + hidden +
            context_size + character_embedding_size + column
        ]
    );
}

__global__ void lstm_gate_backward_kernel(
    const float* gates,
    const float* cell_previous,
    const float* cell_next,
    const float* hidden_gradient,
    const float* cell_gradient,
    float* gate_gradient,
    float* hidden_passthrough,
    float* cell_previous_gradient,
    const int* lengths,
    int timestep,
    int batch,
    int hidden
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= batch * hidden) {
        return;
    }
    const int row = index / hidden;
    const int unit = index - row * hidden;
    if (timestep >= lengths[row]) {
        hidden_passthrough[index] = hidden_gradient[index];
        cell_previous_gradient[index] = cell_gradient[index];
        const std::size_t base =
            static_cast<std::size_t>(row) * 4 * hidden + unit;
        gate_gradient[base] = 0.0f;
        gate_gradient[base + hidden] = 0.0f;
        gate_gradient[base + 2 * hidden] = 0.0f;
        gate_gradient[base + 3 * hidden] = 0.0f;
        return;
    }
    hidden_passthrough[index] = 0.0f;
    const std::size_t base =
        static_cast<std::size_t>(row) * 4 * hidden + unit;
    const float input_gate = gates[base];
    const float forget_gate = gates[base + hidden];
    const float candidate = gates[base + 2 * hidden];
    const float output_gate = gates[base + 3 * hidden];
    const float cell_value = cell_next[index];
    const float tanh_cell = tanhf(cell_value);
    const float total_cell_gradient = cell_gradient[index] +
        hidden_gradient[index] * output_gate *
            (1.0f - tanh_cell * tanh_cell);
    gate_gradient[base] = total_cell_gradient * candidate *
        input_gate * (1.0f - input_gate);
    gate_gradient[base + hidden] = total_cell_gradient *
        cell_previous[index] * forget_gate * (1.0f - forget_gate);
    gate_gradient[base + 2 * hidden] = total_cell_gradient *
        input_gate * (1.0f - candidate * candidate);
    gate_gradient[base + 3 * hidden] = hidden_gradient[index] *
        tanh_cell * output_gate * (1.0f - output_gate);
    cell_previous_gradient[index] =
        total_cell_gradient * forget_gate;
}

__global__ void bias_gradient_kernel(
    const float* gate_gradient,
    float* input_bias_gradient,
    float* recurrent_bias_gradient,
    int batch,
    int gate_size
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= gate_size) {
        return;
    }
    float sum = 0.0f;
    for (int row = 0; row < batch; ++row) {
        sum += gate_gradient[
            static_cast<std::size_t>(row) * gate_size + index
        ];
    }
    input_bias_gradient[index] += sum;
    recurrent_bias_gradient[index] += sum;
}

__global__ void scale_kernel(float* values, std::size_t size, float scale) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        values[index] *= scale;
    }
}

}  // namespace

namespace {

void upload(
    DeviceParameter& parameter,
    std::span<const float> values,
    cudaStream_t stream
);
void download(
    const DeviceParameter& parameter,
    std::vector<float>& values,
    cudaStream_t stream
);
void fill_uniform(
    std::vector<float>& values,
    std::mt19937& random,
    float bound
);
void allocate_network(
    Network& network,
    const RecurrentPolicyBackendConfig& config,
    const std::string& prefix,
    bool trainable,
    int device_id
);
void initialize_network(
    Network& network,
    const RecurrentPolicyBackendConfig& config,
    std::mt19937& random,
    cudaStream_t stream
);
void copy_network(
    const Network& source,
    Network& destination,
    cudaStream_t stream
);
BatchHost make_batch(
    std::span<const Transition* const> transitions,
    bool next_state,
    const RecurrentPolicyBackendConfig& config
);
void forward_network(
    const Network& network,
    const base::Tensor& action_sizes,
    const BatchHost& batch,
    ForwardWorkspace& workspace,
    cublasHandle_t cublas,
    cudaStream_t stream,
    std::size_t character_embedding_size,
    std::size_t opponent_embedding_size
);
void backward_network(
    Network& network,
    const std::vector<int>& actions,
    const std::vector<float>& gradient_q,
    const BatchHost& batch,
    ForwardWorkspace& workspace,
    cublasHandle_t cublas,
    cudaStream_t stream,
    std::size_t character_embedding_size,
    std::size_t opponent_embedding_size
);

base::data::image::Image decode_jpeg(
    std::span<const std::uint8_t> encoded
) {
    if (encoded.empty() ||
        encoded.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid native ResNet JPEG payload");
    }
    int width = 0;
    int height = 0;
    int source_channels = 0;
    unsigned char* pixels = stbi_load_from_memory(
        encoded.data(), static_cast<int>(encoded.size()),
        &width, &height, &source_channels, 3
    );
    if (pixels == nullptr || width <= 0 || height <= 0) {
        if (pixels != nullptr) {
            stbi_image_free(pixels);
        }
        throw std::runtime_error("native ResNet JPEG decode failed");
    }
    base::data::image::Image image(width, height, 3);
    std::memcpy(image.data(), pixels, image.size_bytes());
    stbi_image_free(pixels);
    return image;
}

}  // namespace

struct CppRecurrentPolicyBackend::Implementation {
    explicit Implementation(RecurrentPolicyBackendConfig value)
        : config(std::move(value)),
          random(config.random_seed),
          stream(base::Stream::create(cudaStreamNonBlocking)),
          matchup_wins(config.initial_matchup_wins) {
        validate();
        if (matchup_wins.empty()) {
            matchup_wins.assign(
                config.action_sizes.size() * config.opponent_count, 0U
            );
        }
        check_cuda(
            cudaSetDevice(config.device_id),
            "select native recurrent CUDA device"
        );
        base::StreamGuard stream_guard(stream);
        check_cublas(
            cublasCreate(&cublas),
            "create native recurrent cuBLAS handle"
        );
        check_cublas(
            cublasSetStream(cublas, stream.raw()),
            "set native recurrent cuBLAS stream"
        );
        allocate();
        initialize();
        if (
            config.resume &&
            !config.checkpoint_path.empty() &&
            std::filesystem::is_regular_file(config.checkpoint_path)
        ) {
            load_checkpoint(config.checkpoint_path);
        }
        refresh_stats();
        stream.synchronize();
    }

    ~Implementation() {
        try {
            if (
                config.training_enabled &&
                config.checkpoint_writes_enabled &&
                !config.checkpoint_path.empty()
            ) {
                save_checkpoint(config.checkpoint_path);
            }
        } catch (...) {
        }
        if (cublas != nullptr) {
            cublasDestroy(cublas);
            cublas = nullptr;
        }
    }

    void validate() const {
        if (config.backend != "cpp") {
            throw std::invalid_argument(
                "native recurrent implementation requires backend=cpp"
            );
        }
        if (config.device_id < 0 || config.checkpoint_path.empty()) {
            throw std::invalid_argument(
                "native recurrent checkpoint/device is invalid"
            );
        }
        require_positive(config.visual_size, "visual size");
        require_positive(config.projection_size, "projection size");
        require_positive(config.controller_size, "controller size");
        require_positive(config.temporal_hidden_size, "hidden size");
        require_positive(config.context_size, "context size");
        require_positive(config.strategy_hidden_size, "strategy size");
        require_positive(config.sequence_capacity, "sequence capacity");
        require_positive(config.character_embedding_size,
                         "character embedding size");
        require_positive(config.opponent_embedding_size,
                         "opponent embedding size");
        require_positive(config.opponent_count, "opponent count");
        require_positive(config.replay_capacity, "replay capacity");
        require_positive(
            config.winning_replay_capacity, "winning replay capacity"
        );
        require_positive(
            config.winning_replay_ramp_episodes,
            "winning replay ramp episodes"
        );
        require_positive(config.minimum_replay_size, "minimum replay");
        require_positive(config.batch_size, "batch size");
        require_positive(config.training_updates_per_transition,
                         "updates per transition");
        require_positive(config.target_update_interval,
                         "target update interval");
        require_positive(config.n_step_options, "n-step options");
        if (
            config.action_sizes.empty() ||
            std::any_of(
                config.action_sizes.begin(), config.action_sizes.end(),
                [](std::size_t value) { return value == 0; }
            ) ||
            config.minimum_replay_size > config.replay_capacity ||
            config.batch_size > config.replay_capacity ||
            !(config.gamma > 0.0f && config.gamma <= 1.0f) ||
            config.q_value_limit <= 0.0f ||
            config.terminal_replay_fraction < 0.0f ||
            config.terminal_replay_fraction > 1.0f ||
            config.hard_negative_replay_fraction < 0.0f ||
            config.hard_negative_replay_fraction > 1.0f ||
            config.winning_replay_fraction < 0.0f ||
            config.winning_replay_fraction > 1.0f ||
            config.winning_replay_min_fraction < 0.0f ||
            config.winning_replay_min_fraction >
                config.winning_replay_fraction ||
            (
                !config.initial_matchup_wins.empty() &&
                config.initial_matchup_wins.size() !=
                    config.action_sizes.size() * config.opponent_count
            ) ||
            config.terminal_replay_fraction +
                    config.hard_negative_replay_fraction +
                    config.winning_replay_fraction >
                1.0f ||
            config.hard_negative_q_threshold < 0.0f ||
            config.learning_rate <= 0.0f ||
            config.gradient_clip <= 0.0f
        ) {
            throw std::invalid_argument(
                "invalid native recurrent policy configuration"
            );
        }
        if (config.trainable_resnet &&
            (config.visual_size !=
                    static_cast<std::size_t>(
                        ResNet50FeatureExtractor::kFeatureSize
                    ) ||
             config.resnet50_checkpoint_path.empty() ||
             !std::filesystem::is_regular_file(
                 config.resnet50_checkpoint_path
             ) ||
             config.resnet_learning_rate <= 0.0f ||
             config.resnet_batch_size == 0 ||
             config.resnet_update_interval == 0 ||
             config.resnet_replay_images == 0)) {
            throw std::invalid_argument(
                "invalid native trainable ResNet50 stage4 configuration"
            );
        }
    }

    void allocate() {
        maximum_actions = *std::max_element(
            config.action_sizes.begin(), config.action_sizes.end()
        );
        projection_weight.allocate(
            "projection.weight",
            config.projection_size * config.visual_size,
            false,
            config.device_id
        );
        projection_bias.allocate(
            "projection.bias",
            config.projection_size,
            false,
            config.device_id
        );
        visual_buffer = base::Tensor::zeros(
            {config.visual_size}, base::DType::F32, config.device_id
        );
        projection_buffer = base::Tensor::zeros(
            {config.projection_size}, base::DType::F32, config.device_id
        );
        allocate_network(
            online_a, config, "online_a", true, config.device_id
        );
        allocate_network(
            online_b, config, "online_b", true, config.device_id
        );
        allocate_network(
            target_a, config, "target_a", false, config.device_id
        );
        allocate_network(
            target_b, config, "target_b", false, config.device_id
        );
        std::vector<int> sizes;
        sizes.reserve(config.action_sizes.size());
        for (std::size_t value : config.action_sizes) {
            sizes.push_back(static_cast<int>(value));
        }
        action_sizes_device = base::Tensor::zeros(
            {sizes.size()}, base::DType::I32, config.device_id
        );
        check_cuda(cudaMemcpyAsync(
            int_data(action_sizes_device), sizes.data(),
            sizes.size() * sizeof(int), cudaMemcpyHostToDevice,
            stream.raw()
        ), "upload native action sizes");

        inference_a.allocate(config, 1, config.device_id);
        inference_b.allocate(config, 1, config.device_id);
        training_a.allocate(config, config.batch_size, config.device_id);
        training_b.allocate(config, config.batch_size, config.device_id);
        temporary.allocate(config, config.batch_size, config.device_id);

        std::vector<base::optim::Parameter> parameters;
        std::vector<float*> first_moments;
        std::vector<float*> second_moments;
        for (Network* network : {&online_a, &online_b}) {
            for (DeviceParameter* parameter : network->parameters()) {
                parameters.push_back({
                    parameter->values(),
                    parameter->gradients(),
                    parameter->size,
                });
                first_moments.push_back(parameter->moments1());
                second_moments.push_back(parameter->moments2());
            }
        }
        optimizer = std::make_unique<base::optim::AdamW>(
            std::move(parameters),
            std::move(first_moments),
            std::move(second_moments),
            config.learning_rate,
            kAdamWeightDecay
        );
        if (config.trainable_resnet) {
            visual_encoder = std::make_unique<ResNet50FeatureExtractor>(
                config.device_id, true, config.resnet_batch_size
            );
            inference_visual_encoder =
                std::make_unique<ResNet50FeatureExtractor>(
                    config.device_id, false, 1
                );
            const auto stage4 = visual_encoder->stage4_parameters();
            resnet_first_moments.resize(stage4.size());
            resnet_second_moments.resize(stage4.size());
            std::vector<base::optim::Parameter> parameters;
            std::vector<float*> first_moments;
            std::vector<float*> second_moments;
            parameters.reserve(stage4.size());
            first_moments.reserve(stage4.size());
            second_moments.reserve(stage4.size());
            for (std::size_t index = 0; index < stage4.size(); ++index) {
                resnet_first_moments[index] = base::Tensor::zeros(
                    {stage4[index].size}, base::DType::F32, config.device_id
                );
                resnet_second_moments[index] = base::Tensor::zeros(
                    {stage4[index].size}, base::DType::F32, config.device_id
                );
                parameters.push_back({
                    stage4[index].data,
                    stage4[index].gradient,
                    stage4[index].size,
                });
                first_moments.push_back(
                    float_data(resnet_first_moments[index])
                );
                second_moments.push_back(
                    float_data(resnet_second_moments[index])
                );
            }
            resnet_optimizer = std::make_unique<base::optim::AdamW>(
                std::move(parameters), std::move(first_moments),
                std::move(second_moments), config.resnet_learning_rate,
                kAdamWeightDecay
            );
            const std::size_t projected =
                config.resnet_batch_size * config.projection_size;
            const std::size_t features =
                config.resnet_batch_size * config.visual_size;
            resnet_projected = base::Tensor::zeros(
                {projected}, base::DType::F32, config.device_id
            );
            resnet_projected_gradient = base::Tensor::zeros(
                {projected}, base::DType::F32, config.device_id
            );
            resnet_feature_gradient = base::Tensor::zeros(
                {features}, base::DType::F32, config.device_id
            );
        }
    }

    void initialize() {
        std::vector<float> values(projection_weight.size);
        fill_uniform(
            values,
            random,
            std::sqrt(6.0f / static_cast<float>(
                config.visual_size + config.projection_size
            ))
        );
        upload(projection_weight, values, stream.raw());
        values.assign(projection_bias.size, 0.0f);
        upload(projection_bias, values, stream.raw());
        if (visual_encoder != nullptr) {
            const ResNet50CheckpointReport report =
                visual_encoder->load_checkpoint(
                    config.resnet50_checkpoint_path.string()
                );
            const ResNet50CheckpointReport inference_report =
                inference_visual_encoder->load_checkpoint(
                    config.resnet50_checkpoint_path.string()
                );
            stats.resnet_checkpoint_loaded = report.tensor_count == 320;
            if (inference_report.tensor_count != report.tensor_count ||
                inference_report.checksum != report.checksum) {
                throw std::runtime_error(
                    "native ResNet50 inference/training initialization mismatch"
                );
            }
            const std::filesystem::path stage4_checkpoint =
                resnet_checkpoint_path();
            if (config.resume &&
                std::filesystem::is_regular_file(stage4_checkpoint)) {
                load_resnet_checkpoint(stage4_checkpoint);
            }
            inference_visual_encoder->copy_stage4_from(*visual_encoder);
            refresh_resnet_checksum();
        }
        initialize_network(online_a, config, random, stream.raw());
        initialize_network(online_b, config, random, stream.raw());
        copy_network(online_a, target_a, stream.raw());
        copy_network(online_b, target_b, stream.raw());
        stats.training_updates = config.initial_training_updates;
        stats.backend = "cpp";
        stats.device = "cuda:" + std::to_string(config.device_id) +
            "/cpp-baseai";
        stats.trainable_resnet = visual_encoder != nullptr;
        stats.resnet_checkpoint_loaded = visual_encoder != nullptr ||
            (!config.resnet50_checkpoint_path.empty() &&
             std::filesystem::is_regular_file(
                 config.resnet50_checkpoint_path
             ));
    }

    std::shared_ptr<const SequenceSnapshot> snapshot() const {
        if (current_sequence.empty()) {
            throw std::runtime_error(
                "native recurrent sequence is empty"
            );
        }
        auto result = std::make_shared<SequenceSnapshot>();
        result->length = current_sequence.size();
        result->input_size = config.projection_size + config.controller_size;
        result->values.reserve(result->length * result->input_size);
        for (const auto& step : current_sequence) {
            result->values.insert(
                result->values.end(), step.begin(), step.end()
            );
        }
        return result;
    }

    void observe(
        std::span<const float> visual,
        std::span<const std::uint8_t> jpeg,
        std::uint64_t frame_id,
        std::span<const float> controller,
        std::span<const float> context,
        bool episode_started
    ) {
        if (episode_started) {
            reset();
        }
        std::span<const float> effective_visual = visual;
        if (visual_encoder != nullptr) {
            if (jpeg.empty() || frame_id == 0) {
                throw std::invalid_argument(
                    "native trainable ResNet observation requires JPEG/frame_id"
                );
            }
            if (current_visual == nullptr ||
                current_visual->frame_id != frame_id) {
                base::data::image::Image image = decode_jpeg(jpeg);
                const __half* device_feature =
                    inference_visual_encoder->extract_batch_device(
                        std::span<const base::data::image::Image>(&image, 1),
                        false
                    );
                cached_visual_feature =
                    inference_visual_encoder->download_device_feature(
                        device_feature
                    );
                current_visual = std::make_shared<EncodedVisualFrame>();
                current_visual->frame_id = frame_id;
                current_visual->jpeg.assign(jpeg.begin(), jpeg.end());
            }
            effective_visual = cached_visual_feature;
        }
        if (
            effective_visual.size() != config.visual_size ||
            controller.size() != config.controller_size ||
            context.size() != config.context_size
        ) {
            throw std::invalid_argument(
                "native recurrent observation shape mismatch"
            );
        }
        if (
            !std::all_of(effective_visual.begin(), effective_visual.end(),
                [](float value) { return std::isfinite(value); }) ||
            !std::all_of(controller.begin(), controller.end(),
                [](float value) { return std::isfinite(value); }) ||
            !std::all_of(context.begin(), context.end(),
                [](float value) { return std::isfinite(value); })
        ) {
            throw std::invalid_argument(
                "native recurrent observation is not finite"
            );
        }
        check_cuda(cudaMemcpyAsync(
            float_data(visual_buffer), effective_visual.data(),
            effective_visual.size_bytes(),
            cudaMemcpyHostToDevice, stream.raw()
        ), "upload native recurrent visual feature");
        const int projection_count =
            static_cast<int>(config.projection_size);
        projection_kernel<<<
            (projection_count + 255) / 256, 256, 0, stream.raw()
        >>>(
            float_data(visual_buffer),
            projection_weight.values(),
            projection_bias.values(),
            float_data(projection_buffer),
            static_cast<int>(config.visual_size),
            projection_count
        );
        check_cuda(cudaGetLastError(), "launch native visual projection");
        std::vector<float> projected(config.projection_size);
        check_cuda(cudaMemcpyAsync(
            projected.data(), float_data(projection_buffer),
            projected.size() * sizeof(float), cudaMemcpyDeviceToHost,
            stream.raw()
        ), "download native visual projection");
        stream.synchronize();
        std::vector<__half> step(
            config.projection_size + config.controller_size
        );
        for (std::size_t index = 0; index < projected.size(); ++index) {
            step[index] = __float2half(projected[index]);
        }
        for (std::size_t index = 0; index < controller.size(); ++index) {
            step[config.projection_size + index] =
                __float2half(controller[index]);
        }
        if (current_sequence.size() == config.sequence_capacity) {
            current_sequence.pop_front();
        }
        current_sequence.push_back(std::move(step));
        current_context.assign(context.begin(), context.end());
        stats.sequence_size = current_sequence.size();
    }

    std::vector<float> q_values(
        std::size_t character,
        std::size_t opponent,
        std::size_t action_count
    ) {
        if (
            character >= config.action_sizes.size() ||
            opponent >= config.opponent_count ||
            action_count != config.action_sizes[character]
        ) {
            throw std::invalid_argument(
                "invalid native recurrent identity/action count"
            );
        }
        current_sampling_character = character;
        current_sampling_opponent = opponent;
        refresh_winning_sampling_stats();
        Transition synthetic;
        synthetic.state_sequence = snapshot();
        synthetic.next_sequence = synthetic.state_sequence;
        synthetic.state_context = current_context;
        synthetic.next_context = current_context;
        synthetic.character = character;
        synthetic.opponent = opponent;
        const Transition* pointer = &synthetic;
        const BatchHost batch = make_batch(
            std::span<const Transition* const>(&pointer, 1), false, config
        );
        forward_network(
            online_a, action_sizes_device, batch, inference_a,
            cublas, stream.raw(), config.character_embedding_size,
            config.opponent_embedding_size
        );
        forward_network(
            online_b, action_sizes_device, batch, inference_b,
            cublas, stream.raw(), config.character_embedding_size,
            config.opponent_embedding_size
        );
        std::vector<float> values_a(maximum_actions);
        std::vector<float> values_b(maximum_actions);
        check_cuda(cudaMemcpyAsync(
            values_a.data(), float_data(inference_a.q_values),
            maximum_actions * sizeof(float), cudaMemcpyDeviceToHost,
            stream.raw()
        ), "download native critic A values");
        check_cuda(cudaMemcpyAsync(
            values_b.data(), float_data(inference_b.q_values),
            maximum_actions * sizeof(float), cudaMemcpyDeviceToHost,
            stream.raw()
        ), "download native critic B values");
        stream.synchronize();
        std::vector<float> result(action_count);
        float disagreement = 0.0f;
        for (std::size_t action = 0; action < action_count; ++action) {
            result[action] = std::min(values_a[action], values_b[action]);
            disagreement += std::abs(values_a[action] - values_b[action]);
        }
        stats.last_critic_disagreement =
            disagreement / static_cast<float>(action_count);
        return result;
    }

    void begin_transition(
        std::size_t action,
        std::size_t character,
        std::size_t opponent,
        float selected_q
    ) {
        if (
            character >= config.action_sizes.size() ||
            action >= config.action_sizes[character] ||
            opponent >= config.opponent_count ||
            !std::isfinite(selected_q)
        ) {
            throw std::invalid_argument(
                "invalid native recurrent transition"
            );
        }
        current_sampling_character = character;
        current_sampling_opponent = opponent;
        refresh_winning_sampling_stats();
        active = ActiveTransition{
            .sequence = snapshot(),
            .context = current_context,
            .action = action,
            .character = character,
            .opponent = opponent,
            .selected_q = selected_q,
            .state_jpeg = current_visual,
        };
    }

    Transition combine_pending(std::size_t count) const {
        if (count == 0 || pending_n_step.empty()) {
            throw std::logic_error(
                "cannot combine an empty native n-step transition"
            );
        }
        const Transition& first = pending_n_step.front();
        const Transition& last = pending_n_step[count - 1];
        Transition combined = first;
        combined.next_sequence = last.next_sequence;
        combined.next_context = last.next_context;
        combined.next_action_mask = last.next_action_mask;
        combined.reward = 0.0f;
        combined.duration = 0;
        combined.terminal = false;
        for (std::size_t index = 0; index < count; ++index) {
            const Transition& value = pending_n_step[index];
            combined.reward += std::pow(
                config.gamma,
                static_cast<float>(combined.duration)
            ) * value.reward;
            combined.duration += value.duration;
            combined.terminal = value.terminal;
            if (value.terminal) {
                combined.next_sequence = value.next_sequence;
                combined.next_context = value.next_context;
                combined.next_action_mask = value.next_action_mask;
                break;
            }
        }
        combined.duration = std::max<std::size_t>(1, combined.duration);
        return combined;
    }

    void append_replay(Transition transition) {
        current_episode_transitions.push_back(transition);
        if (replay.size() == config.replay_capacity) {
            replay.pop_front();
        }
        replay.push_back(std::move(transition));
        trim_resnet_replay_images();
        refresh_replay_stats();
    }

    void trim_resnet_replay_images() {
        if (!config.trainable_resnet) {
            return;
        }
        std::size_t retained = 0;
        for (const Transition& value : replay) {
            retained += value.state_jpeg != nullptr &&
                !value.state_jpeg->jpeg.empty();
        }
        for (Transition& value : replay) {
            if (retained <= config.resnet_replay_images) {
                break;
            }
            if (value.state_jpeg != nullptr &&
                !value.state_jpeg->jpeg.empty()) {
                value.state_jpeg->jpeg.clear();
                value.state_jpeg->jpeg.shrink_to_fit();
                --retained;
            }
        }
    }

    std::size_t append_n_step(Transition transition) {
        pending_n_step.push_back(std::move(transition));
        std::size_t appended = 0;
        if (pending_n_step.back().terminal) {
            while (!pending_n_step.empty()) {
                append_replay(combine_pending(pending_n_step.size()));
                pending_n_step.pop_front();
                ++appended;
            }
            return appended;
        }
        if (pending_n_step.size() >= config.n_step_options) {
            append_replay(combine_pending(config.n_step_options));
            pending_n_step.pop_front();
            ++appended;
        }
        return appended;
    }

    void finish_episode(bool successful_episode) {
        if (
            successful_episode &&
            !current_episode_transitions.empty()
        ) {
            const Transition& episode =
                current_episode_transitions.front();
            ++matchup_wins[
                episode.character * config.opponent_count +
                episode.opponent
            ];
            for (Transition transition : current_episode_transitions) {
                transition.successful_episode = true;
                if (
                    winning_replay.size() ==
                    config.winning_replay_capacity
                ) {
                    winning_replay.pop_front();
                }
                winning_replay.push_back(std::move(transition));
            }
            ++stats.winning_episodes;
        }
        current_episode_transitions.clear();
        refresh_replay_stats();
    }

    std::vector<const Transition*> balanced_sample() {
        std::vector<const Transition*> terminal;
        std::vector<const Transition*> nonterminal;
        std::vector<const Transition*> hard_negative;
        std::vector<const Transition*> winning;
        winning.reserve(winning_replay.size());
        for (const Transition& value : winning_replay) {
            winning.push_back(&value);
        }
        for (const Transition& value : replay) {
            if (value.terminal) {
                terminal.push_back(&value);
            } else {
                nonterminal.push_back(&value);
                if (
                    value.selected_q >=
                        config.hard_negative_q_threshold &&
                    value.reward < 0.0f
                ) {
                    hard_negative.push_back(&value);
                }
            }
        }
        refresh_winning_sampling_stats();
        const std::size_t requested_winning =
            stats.effective_winning_replay_fraction > 0.0f
                ? std::max<std::size_t>(1, static_cast<std::size_t>(
                      std::lround(config.batch_size *
                          stats.effective_winning_replay_fraction)
                  ))
                : 0;
        const std::size_t winning_count = winning.empty()
            ? 0 : std::min(config.batch_size, requested_winning);
        const std::size_t requested_terminal =
            config.terminal_replay_fraction > 0.0f
                ? std::max<std::size_t>(1, static_cast<std::size_t>(
                      std::lround(config.batch_size *
                          config.terminal_replay_fraction)
                  ))
                : 0;
        const std::size_t terminal_count = terminal.empty()
            ? 0 : std::min(
                config.batch_size - winning_count,
                requested_terminal
            );
        const std::size_t requested_hard = static_cast<std::size_t>(
            std::lround(config.batch_size *
                config.hard_negative_replay_fraction)
        );
        const std::size_t hard_count = hard_negative.empty()
            ? 0 : std::min(
                config.batch_size - winning_count - terminal_count,
                requested_hard
            );
        const std::size_t regular_count =
            config.batch_size - winning_count - terminal_count - hard_count;
        std::vector<const Transition*> result;
        result.reserve(config.batch_size);
        auto draw = [&](const std::vector<const Transition*>& pool,
                        std::size_t count) {
            if (count == 0) {
                return;
            }
            const auto& source = pool.empty() ? nonterminal : pool;
            const auto& actual_source = source.empty() ? terminal : source;
            if (actual_source.empty()) {
                throw std::logic_error("native replay sampling is empty");
            }
            std::unordered_map<
                std::uint64_t,
                std::vector<const Transition*>
            > buckets;
            for (const Transition* value : actual_source) {
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(value->character) << 32U) |
                    static_cast<std::uint64_t>(value->opponent);
                buckets[key].push_back(value);
            }
            std::vector<const std::vector<const Transition*>*> choices;
            choices.reserve(buckets.size());
            for (const auto& [_, values] : buckets) {
                choices.push_back(&values);
            }
            std::uniform_int_distribution<std::size_t> bucket_distribution(
                0, choices.size() - 1
            );
            for (std::size_t draw_index = 0;
                 draw_index < count; ++draw_index) {
                const auto& bucket = *choices[bucket_distribution(random)];
                std::uniform_int_distribution<std::size_t> item_distribution(
                    0, bucket.size() - 1
                );
                result.push_back(bucket[item_distribution(random)]);
            }
        };
        draw(winning, winning_count);
        draw(terminal, terminal_count);
        draw(hard_negative, hard_count);
        draw(nonterminal, regular_count);
        std::shuffle(result.begin(), result.end(), random);
        stats.last_hard_negative_sample_fraction =
            static_cast<float>(std::count_if(
                result.begin(), result.end(), [&](const Transition* value) {
                    return !value->terminal &&
                        value->selected_q >=
                            config.hard_negative_q_threshold &&
                        value->reward < 0.0f;
                }
            )) / static_cast<float>(result.size());
        stats.last_winning_sample_fraction =
            static_cast<float>(std::count_if(
                result.begin(), result.end(), [](const Transition* value) {
                    return value->successful_episode;
                }
            )) / static_cast<float>(result.size());
        return result;
    }

    void zero_gradients() {
        optimizer->zero_grad(stream.raw());
    }

    void clip_gradients() {
        double squared_norm = 0.0;
        for (Network* network : {&online_a, &online_b}) {
            for (DeviceParameter* parameter : network->parameters()) {
                float norm = 0.0f;
                check_cublas(cublasSnrm2(
                    cublas,
                    static_cast<int>(parameter->size),
                    parameter->gradients(),
                    1,
                    &norm
                ), "measure native recurrent gradient norm");
                squared_norm += static_cast<double>(norm) * norm;
            }
        }
        const double norm = std::sqrt(squared_norm);
        if (norm <= config.gradient_clip || norm <= 0.0) {
            return;
        }
        const float scale = static_cast<float>(
            config.gradient_clip / norm
        );
        for (Network* network : {&online_a, &online_b}) {
            for (DeviceParameter* parameter : network->parameters()) {
                scale_kernel<<<
                    (parameter->size + 255) / 256, 256, 0, stream.raw()
                >>>(
                    parameter->gradients(), parameter->size, scale
                );
            }
        }
        check_cuda(cudaGetLastError(), "clip native recurrent gradients");
    }

    bool train_once() {
        if (
            !config.training_enabled ||
            replay.size() < std::max(
                config.minimum_replay_size, config.batch_size
            )
        ) {
            return false;
        }
        std::vector<const Transition*> sampled = balanced_sample();
        std::size_t resnet_count = 0;
        std::vector<float> replacement_projection;
        const bool update_resnet = visual_encoder != nullptr &&
            (stats.training_updates + 1U) %
                config.resnet_update_interval == 0U;
        if (update_resnet) {
            std::vector<const Transition*> eligible;
            eligible.reserve(replay.size());
            for (const Transition& transition : replay) {
                if (transition.state_jpeg != nullptr &&
                    !transition.state_jpeg->jpeg.empty()) {
                    eligible.push_back(&transition);
                }
            }
            if (!eligible.empty()) {
                std::shuffle(eligible.begin(), eligible.end(), random);
                resnet_count = std::min({
                    config.resnet_batch_size,
                    sampled.size(),
                    eligible.size(),
                });
                std::vector<base::data::image::Image> images;
                images.reserve(resnet_count);
                for (std::size_t row = 0; row < resnet_count; ++row) {
                    sampled[row] = eligible[row];
                    images.push_back(decode_jpeg(
                        eligible[row]->state_jpeg->jpeg
                    ));
                }
                const __half* feature =
                    visual_encoder->extract_batch_device(images, true);
                check_cuda(
                    cudaStreamSynchronize(visual_encoder->stream()),
                    "synchronize native stage4 training forward"
                );
                const int projected_count = static_cast<int>(
                    config.resnet_batch_size * config.projection_size
                );
                projection_half_batch_kernel<<<
                    (projected_count + 255) / 256, 256, 0, stream.raw()
                >>>(
                    feature,
                    projection_weight.values(),
                    projection_bias.values(),
                    float_data(resnet_projected),
                    static_cast<int>(config.resnet_batch_size),
                    static_cast<int>(config.visual_size),
                    static_cast<int>(config.projection_size)
                );
                check_cuda(
                    cudaGetLastError(),
                    "project native trainable ResNet features"
                );
                replacement_projection.resize(
                    resnet_count * config.projection_size
                );
                check_cuda(cudaMemcpyAsync(
                    replacement_projection.data(),
                    float_data(resnet_projected),
                    replacement_projection.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream.raw()
                ), "download native trainable ResNet projection");
                stream.synchronize();
            }
        }
        BatchHost current = make_batch(sampled, false, config);
        for (std::size_t row = 0; row < resnet_count; ++row) {
            const std::size_t step =
                static_cast<std::size_t>(current.lengths[row] - 1);
            float* destination = current.sequence.data() +
                (step * current.batch + row) * current.input;
            std::copy_n(
                replacement_projection.data() +
                    row * config.projection_size,
                config.projection_size,
                destination
            );
        }
        const BatchHost next = make_batch(sampled, true, config);
        zero_gradients();
        forward_network(
            online_a, action_sizes_device, current, training_a,
            cublas, stream.raw(), config.character_embedding_size,
            config.opponent_embedding_size
        );
        forward_network(
            online_b, action_sizes_device, current, training_b,
            cublas, stream.raw(), config.character_embedding_size,
            config.opponent_embedding_size
        );
        std::vector<float> current_a(
            config.batch_size * maximum_actions
        );
        std::vector<float> current_b(current_a.size());
        check_cuda(cudaMemcpyAsync(
            current_a.data(), float_data(training_a.q_values),
            current_a.size() * sizeof(float), cudaMemcpyDeviceToHost,
            stream.raw()
        ), "download current native critic A");
        check_cuda(cudaMemcpyAsync(
            current_b.data(), float_data(training_b.q_values),
            current_b.size() * sizeof(float), cudaMemcpyDeviceToHost,
            stream.raw()
        ), "download current native critic B");

        auto forward_download_next = [&](const Network& network) {
            forward_network(
                network, action_sizes_device, next, temporary,
                cublas, stream.raw(), config.character_embedding_size,
                config.opponent_embedding_size
            );
            std::vector<float> values(
                config.batch_size * maximum_actions
            );
            check_cuda(cudaMemcpyAsync(
                values.data(), float_data(temporary.q_values),
                values.size() * sizeof(float), cudaMemcpyDeviceToHost,
                stream.raw()
            ), "download next native critic values");
            stream.synchronize();
            return values;
        };
        const std::vector<float> online_next_a =
            forward_download_next(online_a);
        const std::vector<float> online_next_b =
            forward_download_next(online_b);
        const std::vector<float> target_next_a =
            forward_download_next(target_a);
        const std::vector<float> target_next_b =
            forward_download_next(target_b);
        stream.synchronize();

        std::vector<int> actions(config.batch_size);
        std::vector<float> gradient_a(config.batch_size);
        std::vector<float> gradient_b(config.batch_size);
        std::vector<float> expected(config.batch_size);
        float predicted_sum = 0.0f;
        float target_sum = 0.0f;
        float bootstrap_sum = 0.0f;
        float gap_sum = 0.0f;
        float absolute_error_sum = 0.0f;
        float disagreement_sum = 0.0f;
        float loss_sum = 0.0f;
        float terminal_predicted_sum = 0.0f;
        float terminal_target_sum = 0.0f;
        std::size_t terminal_count = 0;
        std::size_t clipped_count = 0;
        for (std::size_t row = 0; row < sampled.size(); ++row) {
            const Transition& transition = *sampled[row];
            actions[row] = static_cast<int>(transition.action);
            std::size_t best_action = 0;
            float best_value = -std::numeric_limits<float>::infinity();
            for (std::size_t action = 0;
                 action < config.action_sizes[transition.character];
                 ++action) {
                if (
                    !transition.next_action_mask.empty() &&
                    !transition.next_action_mask[action]
                ) {
                    continue;
                }
                const std::size_t index = row * maximum_actions + action;
                const float value = std::min(
                    online_next_a[index], online_next_b[index]
                );
                if (value > best_value) {
                    best_value = value;
                    best_action = action;
                }
            }
            const std::size_t next_index =
                row * maximum_actions + best_action;
            const float bootstrap = config.conservative_bootstrap
                ? std::min(
                    target_next_a[next_index], target_next_b[next_index]
                )
                : 0.5f * (
                    target_next_a[next_index] + target_next_b[next_index]
                );
            const float raw_target = transition.reward +
                (transition.terminal ? 0.0f :
                    std::pow(
                        config.gamma,
                        static_cast<float>(transition.duration)
                    ) * bootstrap);
            expected[row] = std::clamp(
                raw_target, -config.q_value_limit, config.q_value_limit
            );
            clipped_count += expected[row] != raw_target;
            const std::size_t current_index =
                row * maximum_actions + transition.action;
            const float predicted_a = current_a[current_index];
            const float predicted_b = current_b[current_index];
            const float predicted = 0.5f * (predicted_a + predicted_b);
            const float error_a = predicted_a - expected[row];
            const float error_b = predicted_b - expected[row];
            const float error = predicted - expected[row];
            gradient_a[row] = 0.5f * std::clamp(
                error_a, -1.0f, 1.0f
            ) / static_cast<float>(config.batch_size);
            gradient_b[row] = 0.5f * std::clamp(
                error_b, -1.0f, 1.0f
            ) / static_cast<float>(config.batch_size);
            const auto huber = [](float value) {
                const float absolute = std::abs(value);
                return absolute < 1.0f
                    ? 0.5f * value * value
                    : absolute - 0.5f;
            };
            loss_sum += 0.5f * (huber(error_a) + huber(error_b));
            predicted_sum += predicted;
            target_sum += expected[row];
            bootstrap_sum += bootstrap;
            gap_sum += error;
            absolute_error_sum += std::abs(error);
            disagreement_sum += std::abs(predicted_a - predicted_b);
            if (transition.terminal) {
                ++terminal_count;
                terminal_predicted_sum += predicted;
                terminal_target_sum += expected[row];
            }
        }
        backward_network(
            online_a, actions, gradient_a, current, training_a,
            cublas, stream.raw(), config.character_embedding_size,
            config.opponent_embedding_size
        );
        backward_network(
            online_b, actions, gradient_b, current, training_b,
            cublas, stream.raw(), config.character_embedding_size,
            config.opponent_embedding_size
        );
        if (resnet_count > 0) {
            train_resnet_stage4(current, resnet_count);
        }
        clip_gradients();
        optimizer->step(stream.raw());
        ++stats.training_updates;
        if (stats.training_updates % config.target_update_interval == 0) {
            copy_network(online_a, target_a, stream.raw());
            copy_network(online_b, target_b, stream.raw());
        }
        stream.synchronize();
        const float divisor = static_cast<float>(config.batch_size);
        stats.last_loss = loss_sum / divisor;
        stats.last_predicted_q_mean = predicted_sum / divisor;
        stats.last_td_target_mean = target_sum / divisor;
        stats.last_bootstrap_mean = bootstrap_sum / divisor;
        stats.last_q_target_gap = gap_sum / divisor;
        stats.last_abs_td_error = absolute_error_sum / divisor;
        stats.last_q_target_clipped_fraction =
            static_cast<float>(clipped_count) / divisor;
        stats.last_terminal_sample_fraction =
            static_cast<float>(terminal_count) / divisor;
        stats.last_terminal_predicted_q_mean = terminal_count == 0
            ? 0.0f : terminal_predicted_sum /
                static_cast<float>(terminal_count);
        stats.last_terminal_td_target_mean = terminal_count == 0
            ? 0.0f : terminal_target_sum /
                static_cast<float>(terminal_count);
        stats.last_critic_disagreement = disagreement_sum / divisor;
        refresh_parameter_checksum();
        return true;
    }

    void train_resnet_stage4(
        const BatchHost& current,
        std::size_t count
    ) {
        if (visual_encoder == nullptr || resnet_optimizer == nullptr ||
            count == 0 || count > config.resnet_batch_size) {
            throw std::logic_error("invalid native stage4 training update");
        }
        const std::size_t sequence_values =
            current.steps * current.batch * current.input;
        std::vector<float> gradient_a(sequence_values);
        std::vector<float> gradient_b(sequence_values);
        check_cuda(cudaMemcpyAsync(
            gradient_a.data(), float_data(training_a.d_sequence),
            sequence_values * sizeof(float), cudaMemcpyDeviceToHost,
            stream.raw()
        ), "download critic A visual gradients");
        check_cuda(cudaMemcpyAsync(
            gradient_b.data(), float_data(training_b.d_sequence),
            sequence_values * sizeof(float), cudaMemcpyDeviceToHost,
            stream.raw()
        ), "download critic B visual gradients");
        stream.synchronize();
        std::vector<float> projected_gradient(
            config.resnet_batch_size * config.projection_size, 0.0f
        );
        for (std::size_t row = 0; row < count; ++row) {
            const std::size_t step =
                static_cast<std::size_t>(current.lengths[row] - 1);
            const std::size_t source =
                (step * current.batch + row) * current.input;
            for (std::size_t column = 0;
                 column < config.projection_size; ++column) {
                projected_gradient[row * config.projection_size + column] =
                    gradient_a[source + column] +
                    gradient_b[source + column];
            }
        }
        check_cuda(cudaMemcpyAsync(
            float_data(resnet_projected_gradient),
            projected_gradient.data(),
            projected_gradient.size() * sizeof(float),
            cudaMemcpyHostToDevice,
            stream.raw()
        ), "upload native stage4 projection gradients");
        const std::size_t feature_count =
            config.resnet_batch_size * config.visual_size;
        projection_feature_gradient_kernel<<<
            static_cast<int>((feature_count + 255U) / 256U),
            256, 0, stream.raw()
        >>>(
            float_data(resnet_projected_gradient),
            float_data(resnet_projected),
            projection_weight.values(),
            float_data(resnet_feature_gradient),
            static_cast<int>(config.resnet_batch_size),
            static_cast<int>(config.projection_size),
            static_cast<int>(config.visual_size)
        );
        check_cuda(cudaGetLastError(), "backpropagate native projection");
        stream.synchronize();

        resnet_optimizer->zero_grad(visual_encoder->stream());
        visual_encoder->zero_stage4_grad();
        visual_encoder->backward_stage4(
            float_data(resnet_feature_gradient)
        );
        check_cuda(
            cudaStreamSynchronize(visual_encoder->stream()),
            "synchronize native stage4 backward"
        );
        clip_resnet_gradients();
        resnet_optimizer->step(stream.raw());
        stream.synchronize();
        visual_encoder->sync_stage4_weights();
        check_cuda(
            cudaStreamSynchronize(visual_encoder->stream()),
            "synchronize native stage4 weight copies"
        );
        inference_visual_encoder->copy_stage4_from(*visual_encoder);
        ++stats.resnet_training_updates;
        refresh_resnet_checksum();
        cached_visual_feature.clear();
        current_visual.reset();
    }

    void clip_resnet_gradients() {
        double squared_norm = 0.0;
        const auto parameters = visual_encoder->stage4_parameters();
        for (const auto& parameter : parameters) {
            float norm = 0.0f;
            check_cublas(cublasSnrm2(
                cublas, static_cast<int>(parameter.size),
                parameter.gradient, 1, &norm
            ), "measure native stage4 gradient norm");
            squared_norm += static_cast<double>(norm) * norm;
        }
        const double norm = std::sqrt(squared_norm);
        if (norm <= config.gradient_clip || norm <= 0.0) {
            return;
        }
        const float scale = static_cast<float>(
            config.gradient_clip / norm
        );
        for (const auto& parameter : parameters) {
            scale_kernel<<<
                static_cast<int>((parameter.size + 255U) / 256U),
                256, 0, stream.raw()
            >>>(parameter.gradient, parameter.size, scale);
        }
        check_cuda(cudaGetLastError(), "clip native stage4 gradients");
    }

    RecurrentPolicyCloseResult close_transition(
        std::size_t duration,
        float reward,
        bool terminal,
        bool successful_episode,
        std::span<const std::uint8_t> next_action_mask
    ) {
        bool trained = false;
        if (active.has_value() && config.training_enabled) {
            if (!std::isfinite(reward)) {
                throw std::invalid_argument(
                    "native recurrent reward is not finite"
                );
            }
            if (
                next_action_mask.size() !=
                    config.action_sizes[active->character] ||
                std::none_of(
                    next_action_mask.begin(), next_action_mask.end(),
                    [](std::uint8_t value) { return value != 0; }
                )
            ) {
                throw std::invalid_argument(
                    "invalid native recurrent next-action mask"
                );
            }
            Transition transition{
                .state_sequence = active->sequence,
                .state_context = active->context,
                .next_sequence = snapshot(),
                .next_context = current_context,
                .next_action_mask = std::vector<std::uint8_t>(
                    next_action_mask.begin(), next_action_mask.end()
                ),
                .action = active->action,
                .duration = std::max<std::size_t>(1, duration),
                .reward = reward,
                .terminal = terminal,
                .character = active->character,
                .opponent = active->opponent,
                .selected_q = active->selected_q,
                .state_jpeg = active->state_jpeg,
            };
            const std::size_t appended =
                append_n_step(std::move(transition));
            if (terminal) {
                finish_episode(successful_episode);
            }
            for (std::size_t index = 0;
                 index < appended *
                    config.training_updates_per_transition;
                 ++index) {
                trained = train_once() || trained;
            }
        }
        active.reset();
        refresh_replay_stats();
        return {.stats = stats, .training_performed = trained};
    }

    RecurrentPolicyCloseResult finalize_episode(
        float terminal_reward,
        bool successful_episode,
        std::span<const std::uint8_t> next_action_mask
    ) {
        bool trained = false;
        std::size_t appended = 0;
        if (config.training_enabled) {
            if (!std::isfinite(terminal_reward)) {
                throw std::invalid_argument(
                    "native recurrent terminal reward is not finite"
                );
            }
            const Transition* most_recent = !pending_n_step.empty()
                ? &pending_n_step.back()
                : (
                    !current_episode_transitions.empty()
                        ? &current_episode_transitions.back()
                        : nullptr
                );
            if (most_recent != nullptr) {
                if (
                    next_action_mask.size() !=
                        config.action_sizes[most_recent->character] ||
                    std::none_of(
                        next_action_mask.begin(),
                        next_action_mask.end(),
                        [](std::uint8_t value) { return value != 0; }
                    )
                ) {
                    throw std::invalid_argument(
                        "invalid native recurrent episode action mask"
                    );
                }
                if (!pending_n_step.empty()) {
                    Transition& terminal = pending_n_step.back();
                    terminal.reward += std::pow(
                        config.gamma,
                        static_cast<float>(terminal.duration)
                    ) * terminal_reward;
                    terminal.terminal = true;
                    terminal.next_sequence = snapshot();
                    terminal.next_context = current_context;
                    terminal.next_action_mask.assign(
                        next_action_mask.begin(), next_action_mask.end()
                    );
                    while (!pending_n_step.empty()) {
                        append_replay(
                            combine_pending(pending_n_step.size())
                        );
                        pending_n_step.pop_front();
                        ++appended;
                    }
                } else {
                    Transition terminal = *most_recent;
                    terminal.reward += std::pow(
                        config.gamma,
                        static_cast<float>(terminal.duration)
                    ) * terminal_reward;
                    terminal.terminal = true;
                    terminal.next_sequence = snapshot();
                    terminal.next_context = current_context;
                    terminal.next_action_mask.assign(
                        next_action_mask.begin(), next_action_mask.end()
                    );
                    append_replay(std::move(terminal));
                    ++appended;
                }
            }
            finish_episode(successful_episode);
            for (std::size_t index = 0;
                 index < appended *
                    config.training_updates_per_transition;
                 ++index) {
                trained = train_once() || trained;
            }
        }
        active.reset();
        refresh_replay_stats();
        return {.stats = stats, .training_performed = trained};
    }

    void reset() {
        current_sequence.clear();
        current_context.assign(config.context_size, 0.0f);
        cached_visual_feature.clear();
        current_visual.reset();
        active.reset();
        pending_n_step.clear();
        current_episode_transitions.clear();
        stats.sequence_size = 0;
    }

    void refresh_replay_stats() {
        stats.replay_size = replay.size();
        stats.winning_replay_size = winning_replay.size();
        std::unordered_map<std::uint64_t, bool> matchups;
        for (const Transition& transition : replay) {
            const std::uint64_t key =
                (static_cast<std::uint64_t>(transition.character) << 32U) |
                static_cast<std::uint64_t>(transition.opponent);
            matchups[key] = true;
        }
        stats.matchup_count = matchups.size();
        refresh_winning_sampling_stats();
    }

    void refresh_winning_sampling_stats() {
        const std::uint64_t current_matchup_wins = matchup_wins[
            current_sampling_character * config.opponent_count +
            current_sampling_opponent
        ];
        stats.current_matchup_winning_episodes = current_matchup_wins;
        const float progress = std::min(
            1.0f,
            static_cast<float>(current_matchup_wins) /
                static_cast<float>(
                    config.winning_replay_ramp_episodes
                )
        );
        stats.effective_winning_replay_fraction =
            config.winning_replay_min_fraction +
            (
                config.winning_replay_fraction -
                config.winning_replay_min_fraction
            ) * progress;
    }

    void refresh_parameter_checksum() {
        double checksum = 0.0;
        std::size_t ordinal = 1;
        std::vector<std::vector<float>> samples;
        for (const Network* network : {&online_a, &online_b}) {
            for (const DeviceParameter* parameter : network->parameters()) {
                const std::size_t count =
                    std::min<std::size_t>(256, parameter->size);
                samples.emplace_back(count);
                check_cuda(cudaMemcpyAsync(
                    samples.back().data(), parameter->values(),
                    count * sizeof(float), cudaMemcpyDeviceToHost,
                    stream.raw()
                ), "sample native recurrent checksum");
            }
        }
        stream.synchronize();
        for (const auto& sample : samples) {
            checksum += std::accumulate(
                sample.begin(), sample.end(), 0.0
            ) * static_cast<double>(ordinal++);
        }
        stats.parameter_checksum = checksum;
    }

    void refresh_resnet_checksum() {
        if (visual_encoder == nullptr) {
            stats.resnet_parameter_checksum = 0.0;
            return;
        }
        double checksum = 0.0;
        std::size_t ordinal = 1;
        std::vector<std::vector<float>> samples;
        for (const auto& parameter :
             visual_encoder->stage4_parameters()) {
            const std::size_t count =
                std::min<std::size_t>(256, parameter.size);
            samples.emplace_back(count);
            check_cuda(cudaMemcpyAsync(
                samples.back().data(), parameter.data,
                count * sizeof(float), cudaMemcpyDeviceToHost,
                stream.raw()
            ), "sample native stage4 checksum");
        }
        stream.synchronize();
        for (const auto& sample : samples) {
            checksum += std::accumulate(
                sample.begin(), sample.end(), 0.0
            ) * static_cast<double>(ordinal++);
        }
        stats.resnet_parameter_checksum = checksum;
    }

    void refresh_stats() {
        stats.backend = "cpp";
        stats.device = "cuda:" + std::to_string(config.device_id) +
            "/cpp-baseai";
        stats.trainable_resnet = visual_encoder != nullptr;
        stats.resnet_checkpoint_loaded =
            visual_encoder != nullptr && visual_encoder->loaded() &&
            inference_visual_encoder != nullptr &&
            inference_visual_encoder->loaded();
        refresh_replay_stats();
        refresh_parameter_checksum();
        if (visual_encoder != nullptr) {
            refresh_resnet_checksum();
        }
    }

    std::vector<DeviceParameter*> all_model_parameters() {
        std::vector<DeviceParameter*> result{
            &projection_weight, &projection_bias
        };
        for (Network* network : {
                 &online_a, &online_b, &target_a, &target_b
             }) {
            const auto values = network->parameters();
            result.insert(result.end(), values.begin(), values.end());
        }
        return result;
    }

    std::vector<DeviceParameter*> online_parameters() {
        std::vector<DeviceParameter*> result;
        for (Network* network : {&online_a, &online_b}) {
            const auto values = network->parameters();
            result.insert(result.end(), values.begin(), values.end());
        }
        return result;
    }

    std::vector<float> config_signature() const {
        std::vector<float> result{
            21.0f,
            static_cast<float>(config.visual_size),
            static_cast<float>(config.projection_size),
            static_cast<float>(config.controller_size),
            static_cast<float>(config.temporal_hidden_size),
            static_cast<float>(config.context_size),
            static_cast<float>(config.strategy_hidden_size),
            static_cast<float>(config.sequence_capacity),
            static_cast<float>(config.character_embedding_size),
            static_cast<float>(config.opponent_embedding_size),
            static_cast<float>(config.opponent_count),
            static_cast<float>(config.action_sizes.size()),
        };
        for (std::size_t value : config.action_sizes) {
            result.push_back(static_cast<float>(value));
        }
        return result;
    }

    std::filesystem::path resnet_checkpoint_path() const {
        return config.checkpoint_path.parent_path() /
            "resnet50_stage4_latest.baseai.ckpt";
    }

    void save_resnet_checkpoint(const std::filesystem::path& path) {
        if (visual_encoder == nullptr || resnet_optimizer == nullptr ||
            !config.checkpoint_writes_enabled || path.empty()) {
            return;
        }
        struct HostTensor {
            std::string name;
            std::vector<float> values;
        };
        const auto state_tensors = visual_encoder->stage4_state_tensors();
        const auto parameters = visual_encoder->stage4_parameters();
        if (parameters.size() != resnet_first_moments.size() ||
            parameters.size() != resnet_second_moments.size()) {
            throw std::logic_error("native stage4 optimizer layout mismatch");
        }
        std::vector<HostTensor> model_host;
        model_host.reserve(state_tensors.size());
        for (const auto& tensor : state_tensors) {
            model_host.push_back({tensor.name, std::vector<float>(tensor.size)});
            check_cuda(cudaMemcpyAsync(
                model_host.back().values.data(), tensor.data,
                tensor.size * sizeof(float), cudaMemcpyDeviceToHost,
                stream.raw()
            ), "download native stage4 checkpoint tensor");
        }
        std::vector<HostTensor> optimizer_host;
        optimizer_host.reserve(parameters.size() * 2U);
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            optimizer_host.push_back({
                "adam.m." + parameters[index].name,
                std::vector<float>(parameters[index].size),
            });
            check_cuda(cudaMemcpyAsync(
                optimizer_host.back().values.data(),
                float_data(resnet_first_moments[index]),
                parameters[index].size * sizeof(float),
                cudaMemcpyDeviceToHost, stream.raw()
            ), "download native stage4 Adam first moment");
            optimizer_host.push_back({
                "adam.v." + parameters[index].name,
                std::vector<float>(parameters[index].size),
            });
            check_cuda(cudaMemcpyAsync(
                optimizer_host.back().values.data(),
                float_data(resnet_second_moments[index]),
                parameters[index].size * sizeof(float),
                cudaMemcpyDeviceToHost, stream.raw()
            ), "download native stage4 Adam second moment");
        }
        stream.synchronize();

        base::checkpoint::CheckpointState state;
        state.global_step = stats.training_updates;
        state.best_loss = stats.last_loss;
        state.optimizer.name = "AdamW-stage4";
        const base::optim::AdamWState optimizer_state =
            resnet_optimizer->state();
        state.optimizer.step = optimizer_state.step;
        state.optimizer.learning_rate = optimizer_state.learning_rate;
        state.optimizer.weight_decay = optimizer_state.weight_decay;
        state.optimizer.beta1 = optimizer_state.beta1;
        state.optimizer.beta2 = optimizer_state.beta2;
        state.optimizer.epsilon = optimizer_state.epsilon;
        for (HostTensor& tensor : model_host) {
            state.model.tensors.push_back({
                tensor.name, tensor.values.data(), tensor.values.size(),
                base::checkpoint::DType::Float32,
            });
        }
        for (HostTensor& tensor : optimizer_host) {
            state.optimizer.tensors.push_back({
                tensor.name, tensor.values.data(), tensor.values.size(),
                base::checkpoint::DType::Float32,
            });
        }
        std::filesystem::create_directories(path.parent_path());
        const std::filesystem::path temporary = path.string() + ".tmp";
        if (!base::checkpoint::save_checkpoint(temporary.string(), state)) {
            throw std::runtime_error(
                "failed to save native stage4 checkpoint: " +
                temporary.string()
            );
        }
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error) {
            std::filesystem::remove(path, error);
            error.clear();
            std::filesystem::rename(temporary, path, error);
        }
        if (error) {
            throw std::runtime_error(
                "failed to publish native stage4 checkpoint: " +
                error.message()
            );
        }
    }

    void load_resnet_checkpoint(const std::filesystem::path& path) {
        base::checkpoint::CheckpointState state;
        if (!base::checkpoint::load_checkpoint_metadata(
                path.string(), state
            )) {
            throw std::runtime_error(
                "failed to read native stage4 checkpoint metadata: " +
                path.string()
            );
        }
        auto state_tensors = visual_encoder->stage4_state_tensors();
        const auto parameters = visual_encoder->stage4_parameters();
        std::unordered_map<std::string, ResNet50NamedDeviceTensor*> model_map;
        for (auto& tensor : state_tensors) {
            model_map.emplace(tensor.name, &tensor);
        }
        std::unordered_map<std::string, std::size_t> moment1_map;
        std::unordered_map<std::string, std::size_t> moment2_map;
        for (std::size_t index = 0; index < parameters.size(); ++index) {
            moment1_map.emplace("adam.m." + parameters[index].name, index);
            moment2_map.emplace("adam.v." + parameters[index].name, index);
        }
        std::vector<std::vector<float>> model_host(
            state.model.tensors.size()
        );
        std::vector<std::vector<float>> optimizer_host(
            state.optimizer.tensors.size()
        );
        std::size_t model_seen = 0;
        std::size_t optimizer_seen = 0;
        for (std::size_t index = 0; index < state.model.tensors.size();
             ++index) {
            auto& tensor = state.model.tensors[index];
            const auto found = model_map.find(tensor.name);
            if (found == model_map.end() ||
                tensor.num_elements != found->second->size ||
                tensor.dtype != base::checkpoint::DType::Float32) {
                throw std::runtime_error(
                    "native stage4 checkpoint model layout mismatch: " +
                    tensor.name
                );
            }
            model_host[index].resize(tensor.num_elements);
            tensor.data = model_host[index].data();
            ++model_seen;
        }
        for (std::size_t index = 0; index < state.optimizer.tensors.size();
             ++index) {
            auto& tensor = state.optimizer.tensors[index];
            std::size_t parameter_index = parameters.size();
            if (const auto found = moment1_map.find(tensor.name);
                found != moment1_map.end()) {
                parameter_index = found->second;
            } else if (const auto found = moment2_map.find(tensor.name);
                       found != moment2_map.end()) {
                parameter_index = found->second;
            }
            if (parameter_index == parameters.size() ||
                tensor.num_elements != parameters[parameter_index].size ||
                tensor.dtype != base::checkpoint::DType::Float32) {
                throw std::runtime_error(
                    "native stage4 checkpoint optimizer layout mismatch: " +
                    tensor.name
                );
            }
            optimizer_host[index].resize(tensor.num_elements);
            tensor.data = optimizer_host[index].data();
            ++optimizer_seen;
        }
        if (model_seen != state_tensors.size() ||
            optimizer_seen != parameters.size() * 2U ||
            !base::checkpoint::load_checkpoint_selective(
                path.string(), state
            )) {
            throw std::runtime_error(
                "native stage4 checkpoint is incomplete: " + path.string()
            );
        }
        for (std::size_t index = 0; index < state.model.tensors.size();
             ++index) {
            const auto found = model_map.find(
                state.model.tensors[index].name
            );
            check_cuda(cudaMemcpyAsync(
                found->second->data, model_host[index].data(),
                model_host[index].size() * sizeof(float),
                cudaMemcpyHostToDevice, stream.raw()
            ), "upload native stage4 checkpoint tensor");
        }
        for (std::size_t index = 0; index < state.optimizer.tensors.size();
             ++index) {
            const auto& tensor = state.optimizer.tensors[index];
            if (const auto found = moment1_map.find(tensor.name);
                found != moment1_map.end()) {
                check_cuda(cudaMemcpyAsync(
                    float_data(resnet_first_moments[found->second]),
                    optimizer_host[index].data(),
                    optimizer_host[index].size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream.raw()
                ), "upload native stage4 Adam first moment");
            } else {
                const auto second_found = moment2_map.find(tensor.name);
                check_cuda(cudaMemcpyAsync(
                    float_data(
                        resnet_second_moments[second_found->second]
                    ),
                    optimizer_host[index].data(),
                    optimizer_host[index].size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream.raw()
                ), "upload native stage4 Adam second moment");
            }
        }
        resnet_optimizer->load_state({
            .step = state.optimizer.step,
            .learning_rate = state.optimizer.learning_rate > 0.0f
                ? state.optimizer.learning_rate
                : config.resnet_learning_rate,
            .weight_decay = state.optimizer.weight_decay >= 0.0f
                ? state.optimizer.weight_decay : kAdamWeightDecay,
            .beta1 = state.optimizer.beta1 > 0.0f
                ? state.optimizer.beta1 : 0.9f,
            .beta2 = state.optimizer.beta2 > 0.0f
                ? state.optimizer.beta2 : 0.999f,
            .epsilon = state.optimizer.epsilon > 0.0f
                ? state.optimizer.epsilon : 1.0e-8f,
        });
        stats.resnet_training_updates = state.optimizer.step;
        stream.synchronize();
        visual_encoder->sync_stage4_weights();
        check_cuda(cudaStreamSynchronize(visual_encoder->stream()),
                   "synchronize restored native stage4");
    }

    void save_checkpoint(const std::filesystem::path& path) {
        if (!config.checkpoint_writes_enabled || path.empty()) {
            return;
        }
        std::filesystem::create_directories(path.parent_path());
        struct HostTensor {
            std::string name;
            std::vector<float> values;
        };
        std::vector<HostTensor> model_host;
        model_host.push_back({"cpp.config.signature", config_signature()});
        for (DeviceParameter* parameter : all_model_parameters()) {
            model_host.push_back({parameter->name, {}});
            download(*parameter, model_host.back().values, stream.raw());
        }
        std::vector<HostTensor> optimizer_host;
        for (DeviceParameter* parameter : online_parameters()) {
            optimizer_host.push_back({"adam.m." + parameter->name, {}});
            optimizer_host.back().values.resize(parameter->size);
            check_cuda(cudaMemcpyAsync(
                optimizer_host.back().values.data(), parameter->moments1(),
                parameter->size * sizeof(float), cudaMemcpyDeviceToHost,
                stream.raw()
            ), "download native Adam first moment");
            optimizer_host.push_back({"adam.v." + parameter->name, {}});
            optimizer_host.back().values.resize(parameter->size);
            check_cuda(cudaMemcpyAsync(
                optimizer_host.back().values.data(), parameter->moments2(),
                parameter->size * sizeof(float), cudaMemcpyDeviceToHost,
                stream.raw()
            ), "download native Adam second moment");
        }
        stream.synchronize();
        base::checkpoint::CheckpointState state;
        state.global_step = stats.training_updates;
        state.best_loss = stats.last_loss;
        state.optimizer.name = "AdamW";
        const base::optim::AdamWState optimizer_state = optimizer->state();
        state.optimizer.step = optimizer_state.step;
        state.optimizer.learning_rate = optimizer_state.learning_rate;
        state.optimizer.weight_decay = optimizer_state.weight_decay;
        state.optimizer.beta1 = optimizer_state.beta1;
        state.optimizer.beta2 = optimizer_state.beta2;
        state.optimizer.epsilon = optimizer_state.epsilon;
        for (HostTensor& tensor : model_host) {
            state.model.tensors.push_back({
                tensor.name,
                tensor.values.data(),
                tensor.values.size(),
                base::checkpoint::DType::Float32,
            });
        }
        for (HostTensor& tensor : optimizer_host) {
            state.optimizer.tensors.push_back({
                tensor.name,
                tensor.values.data(),
                tensor.values.size(),
                base::checkpoint::DType::Float32,
            });
        }
        const std::filesystem::path temporary =
            path.string() + ".tmp";
        if (!base::checkpoint::save_checkpoint(
                temporary.string(), state
            )) {
            throw std::runtime_error(
                "failed to save native recurrent checkpoint: " +
                temporary.string()
            );
        }
        std::error_code error;
        std::filesystem::rename(temporary, path, error);
        if (error) {
            std::filesystem::remove(path, error);
            error.clear();
            std::filesystem::rename(temporary, path, error);
        }
        if (error) {
            throw std::runtime_error(
                "failed to publish native recurrent checkpoint: " +
                error.message()
            );
        }
        if (visual_encoder != nullptr) {
            save_resnet_checkpoint(resnet_checkpoint_path());
        }
    }

    void load_checkpoint(const std::filesystem::path& path) {
        base::checkpoint::CheckpointState state;
        if (!base::checkpoint::load_checkpoint_metadata(
                path.string(), state
            )) {
            throw std::runtime_error(
                "failed to read native recurrent checkpoint metadata: " +
                path.string()
            );
        }
        std::unordered_map<std::string, DeviceParameter*> model_map;
        for (DeviceParameter* parameter : all_model_parameters()) {
            model_map.emplace(parameter->name, parameter);
        }
        std::unordered_map<std::string, DeviceParameter*> moment1_map;
        std::unordered_map<std::string, DeviceParameter*> moment2_map;
        for (DeviceParameter* parameter : online_parameters()) {
            moment1_map.emplace("adam.m." + parameter->name, parameter);
            moment2_map.emplace("adam.v." + parameter->name, parameter);
        }
        std::vector<std::vector<float>> model_host(
            state.model.tensors.size()
        );
        std::vector<std::vector<float>> optimizer_host(
            state.optimizer.tensors.size()
        );
        std::vector<float> loaded_signature;
        std::vector<bool> model_seen(model_map.size(), false);
        std::size_t seen_count = 0;
        for (std::size_t index = 0;
             index < state.model.tensors.size(); ++index) {
            auto& tensor = state.model.tensors[index];
            if (tensor.name == "cpp.config.signature") {
                model_host[index].resize(tensor.num_elements);
                tensor.data = model_host[index].data();
                continue;
            }
            const auto found = model_map.find(tensor.name);
            if (found == model_map.end()) {
                tensor.data = nullptr;
                continue;
            }
            if (tensor.num_elements != found->second->size) {
                throw std::runtime_error(
                    "native recurrent checkpoint tensor shape mismatch: " +
                    tensor.name
                );
            }
            model_host[index].resize(tensor.num_elements);
            tensor.data = model_host[index].data();
            ++seen_count;
        }
        for (std::size_t index = 0;
             index < state.optimizer.tensors.size(); ++index) {
            auto& tensor = state.optimizer.tensors[index];
            DeviceParameter* parameter = nullptr;
            if (const auto found = moment1_map.find(tensor.name);
                found != moment1_map.end()) {
                parameter = found->second;
            } else if (const auto found = moment2_map.find(tensor.name);
                       found != moment2_map.end()) {
                parameter = found->second;
            }
            if (parameter == nullptr) {
                tensor.data = nullptr;
                continue;
            }
            if (tensor.num_elements != parameter->size) {
                throw std::runtime_error(
                    "native recurrent optimizer tensor shape mismatch: " +
                    tensor.name
                );
            }
            optimizer_host[index].resize(tensor.num_elements);
            tensor.data = optimizer_host[index].data();
        }
        if (!base::checkpoint::load_checkpoint_selective(
                path.string(), state
            )) {
            throw std::runtime_error(
                "failed to load native recurrent checkpoint: " +
                path.string()
            );
        }
        const std::vector<float> expected_signature = config_signature();
        bool signature_found = false;
        for (std::size_t index = 0;
             index < state.model.tensors.size(); ++index) {
            const auto& tensor = state.model.tensors[index];
            if (tensor.name == "cpp.config.signature") {
                signature_found = true;
                if (model_host[index] != expected_signature) {
                    throw std::runtime_error(
                        "native recurrent checkpoint architecture mismatch"
                    );
                }
                continue;
            }
            const auto found = model_map.find(tensor.name);
            if (found == model_map.end()) {
                continue;
            }
            upload(*found->second, model_host[index], stream.raw());
        }
        if (!signature_found || seen_count != model_map.size()) {
            throw std::runtime_error(
                "native recurrent checkpoint is incomplete"
            );
        }
        for (std::size_t index = 0;
             index < state.optimizer.tensors.size(); ++index) {
            const auto& tensor = state.optimizer.tensors[index];
            if (const auto found = moment1_map.find(tensor.name);
                found != moment1_map.end()) {
                check_cuda(cudaMemcpyAsync(
                    found->second->moments1(), optimizer_host[index].data(),
                    optimizer_host[index].size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream.raw()
                ), "upload native Adam first moment");
            } else if (const auto found = moment2_map.find(tensor.name);
                       found != moment2_map.end()) {
                check_cuda(cudaMemcpyAsync(
                    found->second->moments2(), optimizer_host[index].data(),
                    optimizer_host[index].size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream.raw()
                ), "upload native Adam second moment");
            }
        }
        base::optim::AdamWState optimizer_state{
            .step = state.optimizer.step,
            .learning_rate = state.optimizer.learning_rate > 0.0f
                ? state.optimizer.learning_rate : config.learning_rate,
            .weight_decay = state.optimizer.weight_decay >= 0.0f
                ? state.optimizer.weight_decay : kAdamWeightDecay,
            .beta1 = state.optimizer.beta1 > 0.0f
                ? state.optimizer.beta1 : 0.9f,
            .beta2 = state.optimizer.beta2 > 0.0f
                ? state.optimizer.beta2 : 0.999f,
            .epsilon = state.optimizer.epsilon > 0.0f
                ? state.optimizer.epsilon : 1.0e-8f,
        };
        optimizer->load_state(optimizer_state);
        stats.training_updates = state.global_step;
        stats.last_loss = state.best_loss;
        stats.checkpoint_loaded = true;
        stats.checkpoint_migrated =
            state.optimizer.tensors.empty();
        stream.synchronize();
    }

    RecurrentPolicyBackendConfig config;
    RecurrentPolicyBackendStats stats;
    std::mt19937 random;
    base::Stream stream;
    cublasHandle_t cublas = nullptr;
    std::size_t maximum_actions = 0;
    DeviceParameter projection_weight;
    DeviceParameter projection_bias;
    base::Tensor visual_buffer;
    base::Tensor projection_buffer;
    base::Tensor action_sizes_device;
    Network online_a;
    Network online_b;
    Network target_a;
    Network target_b;
    std::unique_ptr<base::optim::AdamW> optimizer;
    std::unique_ptr<ResNet50FeatureExtractor> visual_encoder;
    std::unique_ptr<ResNet50FeatureExtractor> inference_visual_encoder;
    std::unique_ptr<base::optim::AdamW> resnet_optimizer;
    std::vector<base::Tensor> resnet_first_moments;
    std::vector<base::Tensor> resnet_second_moments;
    base::Tensor resnet_projected;
    base::Tensor resnet_projected_gradient;
    base::Tensor resnet_feature_gradient;
    ForwardWorkspace inference_a;
    ForwardWorkspace inference_b;
    ForwardWorkspace training_a;
    ForwardWorkspace training_b;
    ForwardWorkspace temporary;
    std::deque<std::vector<__half>> current_sequence;
    std::shared_ptr<EncodedVisualFrame> current_visual;
    std::vector<float> cached_visual_feature;
    std::vector<float> current_context;
    std::optional<ActiveTransition> active;
    std::deque<Transition> pending_n_step;
    std::deque<Transition> replay;
    std::vector<Transition> current_episode_transitions;
    std::deque<Transition> winning_replay;
    std::vector<std::uint64_t> matchup_wins;
    std::size_t current_sampling_character = 0;
    std::size_t current_sampling_opponent = 0;
};

CppRecurrentPolicyBackend::CppRecurrentPolicyBackend(
    RecurrentPolicyBackendConfig config
) : implementation_(
        std::make_unique<Implementation>(std::move(config))
    ) {}

CppRecurrentPolicyBackend::~CppRecurrentPolicyBackend() = default;

void CppRecurrentPolicyBackend::observe(
    std::span<const float> visual,
    std::span<const std::uint8_t> jpeg,
    std::uint64_t frame_id,
    std::span<const float> controller,
    std::span<const float> context,
    bool episode_started
) {
    implementation_->observe(
        visual, jpeg, frame_id, controller, context, episode_started
    );
}

std::vector<float> CppRecurrentPolicyBackend::q_values(
    std::size_t character,
    std::size_t opponent,
    std::size_t action_count
) {
    return implementation_->q_values(
        character, opponent, action_count
    );
}

void CppRecurrentPolicyBackend::begin_transition(
    std::size_t action,
    std::size_t character,
    std::size_t opponent,
    float selected_q
) {
    implementation_->begin_transition(
        action, character, opponent, selected_q
    );
}

RecurrentPolicyCloseResult CppRecurrentPolicyBackend::close_transition(
    std::size_t duration,
    float reward,
    bool terminal,
    bool successful_episode,
    std::span<const std::uint8_t> next_action_mask
) {
    return implementation_->close_transition(
        duration, reward, terminal, successful_episode, next_action_mask
    );
}

RecurrentPolicyCloseResult CppRecurrentPolicyBackend::finalize_episode(
    float terminal_reward,
    bool successful_episode,
    std::span<const std::uint8_t> next_action_mask
) {
    return implementation_->finalize_episode(
        terminal_reward, successful_episode, next_action_mask
    );
}

void CppRecurrentPolicyBackend::reset() {
    implementation_->reset();
}

void CppRecurrentPolicyBackend::set_training_enabled(bool enabled) {
    implementation_->config.training_enabled = enabled;
    implementation_->config.checkpoint_writes_enabled = enabled;
    implementation_->reset();
}

void CppRecurrentPolicyBackend::save(
    const std::filesystem::path& path,
    bool
) {
    implementation_->save_checkpoint(
        path.empty() ? implementation_->config.checkpoint_path : path
    );
}

const RecurrentPolicyBackendStats&
CppRecurrentPolicyBackend::stats() const {
    return implementation_->stats;
}

const RecurrentPolicyBackendConfig&
CppRecurrentPolicyBackend::config() const {
    return implementation_->config;
}

}  // namespace gameagent

namespace gameagent {
namespace {

void row_gemm_abt(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    int rows,
    int outputs,
    int inner,
    float beta = 0.0f
) {
    const float alpha = 1.0f;
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            outputs,
            rows,
            inner,
            &alpha,
            b,
            inner,
            a,
            inner,
            &beta,
            c,
            outputs
        ),
        "native recurrent row GEMM A*B^T"
    );
}

void row_gemm(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    int rows,
    int outputs,
    int inner,
    float beta = 0.0f
) {
    const float alpha = 1.0f;
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            outputs,
            rows,
            inner,
            &alpha,
            b,
            outputs,
            a,
            inner,
            &beta,
            c,
            outputs
        ),
        "native recurrent row GEMM A*B"
    );
}

void row_gemm_at_b(
    cublasHandle_t handle,
    const float* a,
    const float* b,
    float* c,
    int rows,
    int a_width,
    int b_width,
    float beta = 1.0f
) {
    const float alpha = 1.0f;
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            b_width,
            a_width,
            rows,
            &alpha,
            b,
            b_width,
            a,
            a_width,
            &beta,
            c,
            b_width
        ),
        "native recurrent row GEMM A^T*B"
    );
}

void upload(
    DeviceParameter& parameter,
    std::span<const float> values,
    cudaStream_t stream
) {
    if (values.size() != parameter.size) {
        throw std::invalid_argument(
            "native recurrent parameter upload size mismatch: " +
            parameter.name
        );
    }
    check_cuda(
        cudaMemcpyAsync(
            parameter.values(),
            values.data(),
            values.size_bytes(),
            cudaMemcpyHostToDevice,
            stream
        ),
        "upload native recurrent parameter"
    );
}

void download(
    const DeviceParameter& parameter,
    std::vector<float>& values,
    cudaStream_t stream
) {
    values.resize(parameter.size);
    check_cuda(
        cudaMemcpyAsync(
            values.data(),
            parameter.values(),
            values.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        ),
        "download native recurrent parameter"
    );
}

void fill_uniform(
    std::vector<float>& values,
    std::mt19937& random,
    float bound
) {
    std::uniform_real_distribution<float> distribution(-bound, bound);
    for (float& value : values) {
        value = distribution(random);
    }
}

void fill_normal(
    std::vector<float>& values,
    std::mt19937& random,
    float standard_deviation
) {
    std::normal_distribution<float> distribution(
        0.0f, standard_deviation
    );
    for (float& value : values) {
        value = distribution(random);
    }
}

void allocate_network(
    Network& network,
    const RecurrentPolicyBackendConfig& config,
    const std::string& prefix,
    bool trainable,
    int device_id
) {
    const std::size_t input =
        config.projection_size + config.controller_size;
    const std::size_t hidden = config.temporal_hidden_size;
    const std::size_t joined = hidden + config.context_size +
        config.character_embedding_size +
        config.opponent_embedding_size;
    const std::size_t character_count = config.action_sizes.size();
    const std::size_t maximum_actions = *std::max_element(
        config.action_sizes.begin(), config.action_sizes.end()
    );
    network.input_weight.allocate(
        prefix + ".lstm.weight_ih", 4 * hidden * input,
        trainable, device_id
    );
    network.recurrent_weight.allocate(
        prefix + ".lstm.weight_hh", 4 * hidden * hidden,
        trainable, device_id
    );
    network.input_bias.allocate(
        prefix + ".lstm.bias_ih", 4 * hidden, trainable, device_id
    );
    network.recurrent_bias.allocate(
        prefix + ".lstm.bias_hh", 4 * hidden, trainable, device_id
    );
    network.character_embedding.allocate(
        prefix + ".character_embedding",
        character_count * config.character_embedding_size,
        trainable, device_id
    );
    network.opponent_embedding.allocate(
        prefix + ".opponent_embedding",
        config.opponent_count * config.opponent_embedding_size,
        trainable, device_id
    );
    network.strategy_weight.allocate(
        prefix + ".strategy.weight",
        config.strategy_hidden_size * joined,
        trainable, device_id
    );
    network.strategy_bias.allocate(
        prefix + ".strategy.bias",
        config.strategy_hidden_size,
        trainable, device_id
    );
    network.head_weight.allocate(
        prefix + ".heads.weight",
        character_count * maximum_actions *
            config.strategy_hidden_size,
        trainable, device_id
    );
    network.head_bias.allocate(
        prefix + ".heads.bias",
        character_count * maximum_actions,
        trainable, device_id
    );
}

void initialize_network(
    Network& network,
    const RecurrentPolicyBackendConfig& config,
    std::mt19937& random,
    cudaStream_t stream
) {
    const std::size_t input =
        config.projection_size + config.controller_size;
    const std::size_t hidden = config.temporal_hidden_size;
    const std::size_t joined = hidden + config.context_size +
        config.character_embedding_size +
        config.opponent_embedding_size;
    std::vector<float> values;

    values.resize(network.input_weight.size);
    fill_uniform(
        values, random,
        std::sqrt(6.0f / static_cast<float>(input + 4 * hidden))
    );
    upload(network.input_weight, values, stream);

    values.resize(network.recurrent_weight.size);
    fill_uniform(
        values, random,
        std::sqrt(6.0f / static_cast<float>(hidden + 4 * hidden))
    );
    upload(network.recurrent_weight, values, stream);

    values.assign(4 * hidden, 0.0f);
    std::fill(
        values.begin() + static_cast<std::ptrdiff_t>(hidden),
        values.begin() + static_cast<std::ptrdiff_t>(2 * hidden),
        1.0f
    );
    upload(network.input_bias, values, stream);
    values.assign(4 * hidden, 0.0f);
    upload(network.recurrent_bias, values, stream);

    values.resize(network.character_embedding.size);
    fill_normal(
        values, random,
        1.0f / std::sqrt(
            static_cast<float>(config.character_embedding_size)
        )
    );
    upload(network.character_embedding, values, stream);

    values.resize(network.opponent_embedding.size);
    fill_normal(
        values, random,
        1.0f / std::sqrt(
            static_cast<float>(config.opponent_embedding_size)
        )
    );
    upload(network.opponent_embedding, values, stream);

    values.resize(network.strategy_weight.size);
    fill_uniform(
        values, random,
        std::sqrt(6.0f / static_cast<float>(joined))
    );
    upload(network.strategy_weight, values, stream);
    values.assign(network.strategy_bias.size, 0.0f);
    upload(network.strategy_bias, values, stream);

    values.resize(network.head_weight.size);
    fill_uniform(
        values, random,
        std::sqrt(3.0f /
            static_cast<float>(config.strategy_hidden_size))
    );
    upload(network.head_weight, values, stream);
    values.assign(network.head_bias.size, 0.0f);
    upload(network.head_bias, values, stream);
}

void copy_network(
    const Network& source,
    Network& destination,
    cudaStream_t stream
) {
    const auto source_parameters = source.parameters();
    const auto destination_parameters = destination.parameters();
    if (source_parameters.size() != destination_parameters.size()) {
        throw std::logic_error("native recurrent network copy mismatch");
    }
    for (std::size_t index = 0; index < source_parameters.size(); ++index) {
        if (source_parameters[index]->size !=
            destination_parameters[index]->size) {
            throw std::logic_error(
                "native recurrent network tensor copy mismatch"
            );
        }
        check_cuda(
            cudaMemcpyAsync(
                destination_parameters[index]->values(),
                source_parameters[index]->values(),
                source_parameters[index]->size * sizeof(float),
                cudaMemcpyDeviceToDevice,
                stream
            ),
            "copy native recurrent network"
        );
    }
}

BatchHost make_batch(
    std::span<const Transition* const> transitions,
    bool next_state,
    const RecurrentPolicyBackendConfig& config
) {
    BatchHost batch;
    batch.batch = transitions.size();
    batch.input = config.projection_size + config.controller_size;
    batch.steps = 1;
    for (const Transition* transition : transitions) {
        const auto& sequence = next_state
            ? transition->next_sequence
            : transition->state_sequence;
        if (!sequence || sequence->length == 0 ||
            sequence->input_size != batch.input) {
            throw std::runtime_error(
                "native recurrent replay contains an invalid sequence"
            );
        }
        batch.steps = std::max(batch.steps, sequence->length);
    }
    batch.sequence.assign(
        batch.steps * batch.batch * batch.input, 0.0f
    );
    batch.lengths.resize(batch.batch);
    batch.context.resize(batch.batch * config.context_size);
    batch.characters.resize(batch.batch);
    batch.opponents.resize(batch.batch);
    for (std::size_t row = 0; row < batch.batch; ++row) {
        const Transition& transition = *transitions[row];
        const auto& sequence = next_state
            ? transition.next_sequence
            : transition.state_sequence;
        const auto& context = next_state
            ? transition.next_context
            : transition.state_context;
        batch.lengths[row] = static_cast<int>(sequence->length);
        batch.characters[row] = static_cast<int>(transition.character);
        batch.opponents[row] = static_cast<int>(transition.opponent);
        std::copy(
            context.begin(), context.end(),
            batch.context.begin() +
                static_cast<std::ptrdiff_t>(row * config.context_size)
        );
        for (std::size_t step = 0; step < sequence->length; ++step) {
            const __half* source = sequence->values.data() +
                step * batch.input;
            float* destination = batch.sequence.data() +
                (step * batch.batch + row) * batch.input;
            for (std::size_t column = 0; column < batch.input; ++column) {
                destination[column] = __half2float(source[column]);
            }
        }
    }
    return batch;
}

void upload_batch(
    const BatchHost& batch,
    ForwardWorkspace& workspace,
    cudaStream_t stream
) {
    if (
        batch.batch > workspace.batch_capacity ||
        batch.steps > workspace.step_capacity ||
        batch.input != workspace.input_size
    ) {
        throw std::invalid_argument(
            "native recurrent forward workspace capacity exceeded"
        );
    }
    check_cuda(cudaMemcpyAsync(
        float_data(workspace.sequence), batch.sequence.data(),
        batch.sequence.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream
    ), "upload recurrent batch sequence");
    check_cuda(cudaMemcpyAsync(
        int_data(workspace.lengths), batch.lengths.data(),
        batch.lengths.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream
    ), "upload recurrent batch lengths");
    check_cuda(cudaMemcpyAsync(
        float_data(workspace.context), batch.context.data(),
        batch.context.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream
    ), "upload recurrent batch context");
    check_cuda(cudaMemcpyAsync(
        int_data(workspace.characters), batch.characters.data(),
        batch.characters.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream
    ), "upload recurrent batch characters");
    check_cuda(cudaMemcpyAsync(
        int_data(workspace.opponents), batch.opponents.data(),
        batch.opponents.size() * sizeof(int),
        cudaMemcpyHostToDevice, stream
    ), "upload recurrent batch opponents");
}

void forward_network(
    const Network& network,
    const base::Tensor& action_sizes,
    const BatchHost& batch,
    ForwardWorkspace& workspace,
    cublasHandle_t cublas,
    cudaStream_t stream,
    std::size_t character_embedding_size,
    std::size_t opponent_embedding_size
) {
    upload_batch(batch, workspace, stream);
    const int batch_size = static_cast<int>(batch.batch);
    const int steps = static_cast<int>(batch.steps);
    const int input = static_cast<int>(workspace.input_size);
    const int hidden = static_cast<int>(workspace.hidden_size);
    const int gates = 4 * hidden;
    check_cuda(cudaMemsetAsync(
        float_data(workspace.hidden), 0,
        (batch.steps + 1) * batch.batch * hidden * sizeof(float), stream
    ), "zero recurrent hidden workspace");
    check_cuda(cudaMemsetAsync(
        float_data(workspace.cell), 0,
        (batch.steps + 1) * batch.batch * hidden * sizeof(float), stream
    ), "zero recurrent cell workspace");

    for (int step = 0; step < steps; ++step) {
        const float* input_step = float_data(workspace.sequence) +
            static_cast<std::size_t>(step) * batch_size * input;
        const float* hidden_previous = float_data(workspace.hidden) +
            static_cast<std::size_t>(step) * batch_size * hidden;
        const float* cell_previous = float_data(workspace.cell) +
            static_cast<std::size_t>(step) * batch_size * hidden;
        float* hidden_next = float_data(workspace.hidden) +
            static_cast<std::size_t>(step + 1) * batch_size * hidden;
        float* cell_next = float_data(workspace.cell) +
            static_cast<std::size_t>(step + 1) * batch_size * hidden;
        float* gate_step = float_data(workspace.gates) +
            static_cast<std::size_t>(step) * batch_size * gates;
        row_gemm_abt(
            cublas, input_step, network.input_weight.values(), gate_step,
            batch_size, gates, input
        );
        row_gemm_abt(
            cublas, hidden_previous,
            network.recurrent_weight.values(), gate_step,
            batch_size, gates, hidden, 1.0f
        );
        const int count = batch_size * hidden;
        lstm_state_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
            gate_step,
            network.input_bias.values(),
            network.recurrent_bias.values(),
            hidden_previous,
            cell_previous,
            hidden_next,
            cell_next,
            int_data(workspace.lengths),
            step,
            batch_size,
            hidden
        );
        check_cuda(cudaGetLastError(), "launch native LSTM state kernel");
    }

    const int joined_count =
        batch_size * static_cast<int>(workspace.joined_size);
    joined_kernel<<<(joined_count + 255) / 256, 256, 0, stream>>>(
        float_data(workspace.hidden),
        int_data(workspace.lengths),
        float_data(workspace.context),
        int_data(workspace.characters),
        int_data(workspace.opponents),
        network.character_embedding.values(),
        network.opponent_embedding.values(),
        float_data(workspace.joined),
        batch_size,
        hidden,
        static_cast<int>(workspace.context_size),
        static_cast<int>(character_embedding_size),
        static_cast<int>(opponent_embedding_size),
        static_cast<int>(workspace.joined_size)
    );
    check_cuda(cudaGetLastError(), "launch recurrent joined-state kernel");

    row_gemm_abt(
        cublas,
        float_data(workspace.joined),
        network.strategy_weight.values(),
        float_data(workspace.strategy_pre),
        batch_size,
        static_cast<int>(workspace.strategy_size),
        static_cast<int>(workspace.joined_size)
    );
    const int strategy_count =
        batch_size * static_cast<int>(workspace.strategy_size);
    bias_relu_kernel<<<
        (strategy_count + 255) / 256, 256, 0, stream
    >>>(
        float_data(workspace.strategy_pre),
        network.strategy_bias.values(),
        float_data(workspace.strategy),
        batch_size,
        static_cast<int>(workspace.strategy_size)
    );
    check_cuda(cudaGetLastError(), "launch recurrent strategy kernel");

    const int q_count = batch_size * static_cast<int>(workspace.max_actions);
    q_head_kernel<<<(q_count + 255) / 256, 256, 0, stream>>>(
        float_data(workspace.strategy),
        int_data(workspace.characters),
        int_data(action_sizes),
        network.head_weight.values(),
        network.head_bias.values(),
        float_data(workspace.q_values),
        batch_size,
        static_cast<int>(workspace.strategy_size),
        static_cast<int>(workspace.max_actions)
    );
    check_cuda(cudaGetLastError(), "launch recurrent action-head kernel");
}

void backward_network(
    Network& network,
    const std::vector<int>& actions,
    const std::vector<float>& gradient_q,
    const BatchHost& batch,
    ForwardWorkspace& workspace,
    cublasHandle_t cublas,
    cudaStream_t stream,
    std::size_t character_embedding_size,
    std::size_t opponent_embedding_size
) {
    base::Tensor device_actions = base::Tensor::zeros(
        {batch.batch}, base::DType::I32, network.input_weight.value.device_id()
    );
    check_cuda(cudaMemcpyAsync(
        int_data(device_actions), actions.data(),
        actions.size() * sizeof(int), cudaMemcpyHostToDevice, stream
    ), "upload recurrent selected actions");
    check_cuda(cudaMemcpyAsync(
        float_data(workspace.gradient_q), gradient_q.data(),
        gradient_q.size() * sizeof(float), cudaMemcpyHostToDevice, stream
    ), "upload recurrent Q gradients");
    check_cuda(cudaMemsetAsync(
        float_data(workspace.d_sequence), 0,
        batch.steps * batch.batch * workspace.input_size * sizeof(float),
        stream
    ), "zero recurrent input gradients");

    const int batch_size = static_cast<int>(batch.batch);
    const int hidden = static_cast<int>(workspace.hidden_size);
    const int strategy = static_cast<int>(workspace.strategy_size);
    const int joined = static_cast<int>(workspace.joined_size);
    const int strategy_count = batch_size * strategy;
    head_backward_kernel<<<
        (strategy_count + 255) / 256, 256, 0, stream
    >>>(
        float_data(workspace.strategy),
        float_data(workspace.strategy_pre),
        int_data(workspace.characters),
        int_data(device_actions),
        float_data(workspace.gradient_q),
        network.head_weight.values(),
        network.head_weight.gradients(),
        network.head_bias.gradients(),
        float_data(workspace.d_strategy_pre),
        batch_size,
        strategy,
        static_cast<int>(workspace.max_actions)
    );
    check_cuda(cudaGetLastError(), "launch recurrent head backward kernel");

    const int strategy_weights = strategy * joined;
    strategy_weight_gradient_kernel<<<
        (strategy_weights + 255) / 256, 256, 0, stream
    >>>(
        float_data(workspace.d_strategy_pre),
        float_data(workspace.joined),
        network.strategy_weight.gradients(),
        batch_size,
        strategy,
        joined
    );
    strategy_bias_gradient_kernel<<<
        (strategy + 255) / 256, 256, 0, stream
    >>>(
        float_data(workspace.d_strategy_pre),
        network.strategy_bias.gradients(),
        batch_size,
        strategy
    );
    check_cuda(cudaGetLastError(), "launch recurrent strategy gradients");

    row_gemm(
        cublas,
        float_data(workspace.d_strategy_pre),
        network.strategy_weight.values(),
        float_data(workspace.d_joined),
        batch_size,
        joined,
        strategy
    );
    const int recurrent_feature_count = hidden +
        static_cast<int>(character_embedding_size) +
        static_cast<int>(opponent_embedding_size);
    const int recurrent_gradient_count =
        batch_size * recurrent_feature_count;
    embedding_and_hidden_gradient_kernel<<<
        (recurrent_gradient_count + 255) / 256, 256, 0, stream
    >>>(
        float_data(workspace.d_joined),
        int_data(workspace.characters),
        int_data(workspace.opponents),
        network.character_embedding.gradients(),
        network.opponent_embedding.gradients(),
        float_data(workspace.d_hidden),
        batch_size,
        hidden,
        static_cast<int>(workspace.context_size),
        static_cast<int>(character_embedding_size),
        static_cast<int>(opponent_embedding_size),
        joined
    );
    check_cuda(cudaGetLastError(), "launch recurrent embedding gradients");
    check_cuda(cudaMemsetAsync(
        float_data(workspace.d_cell), 0,
        batch.batch * hidden * sizeof(float), stream
    ), "zero recurrent cell gradient");

    const int input = static_cast<int>(workspace.input_size);
    const int gate_size = 4 * hidden;
    for (int step = static_cast<int>(batch.steps) - 1;
         step >= 0; --step) {
        const float* gate_step = float_data(workspace.gates) +
            static_cast<std::size_t>(step) * batch_size * gate_size;
        const float* cell_previous = float_data(workspace.cell) +
            static_cast<std::size_t>(step) * batch_size * hidden;
        const float* cell_next = float_data(workspace.cell) +
            static_cast<std::size_t>(step + 1) * batch_size * hidden;
        const float* hidden_previous = float_data(workspace.hidden) +
            static_cast<std::size_t>(step) * batch_size * hidden;
        const float* input_step = float_data(workspace.sequence) +
            static_cast<std::size_t>(step) * batch_size * input;
        const int count = batch_size * hidden;
        lstm_gate_backward_kernel<<<
            (count + 255) / 256, 256, 0, stream
        >>>(
            gate_step,
            cell_previous,
            cell_next,
            float_data(workspace.d_hidden),
            float_data(workspace.d_cell),
            float_data(workspace.d_gates),
            float_data(workspace.d_hidden_previous),
            float_data(workspace.d_cell_previous),
            int_data(workspace.lengths),
            step,
            batch_size,
            hidden
        );
        check_cuda(cudaGetLastError(), "launch native LSTM backward kernel");

        row_gemm_at_b(
            cublas,
            float_data(workspace.d_gates),
            input_step,
            network.input_weight.gradients(),
            batch_size,
            gate_size,
            input
        );
        row_gemm(
            cublas,
            float_data(workspace.d_gates),
            network.input_weight.values(),
            float_data(workspace.d_sequence) +
                static_cast<std::size_t>(step) * batch_size * input,
            batch_size,
            input,
            gate_size
        );
        row_gemm_at_b(
            cublas,
            float_data(workspace.d_gates),
            hidden_previous,
            network.recurrent_weight.gradients(),
            batch_size,
            gate_size,
            hidden
        );
        bias_gradient_kernel<<<
            (gate_size + 255) / 256, 256, 0, stream
        >>>(
            float_data(workspace.d_gates),
            network.input_bias.gradients(),
            network.recurrent_bias.gradients(),
            batch_size,
            gate_size
        );
        row_gemm(
            cublas,
            float_data(workspace.d_gates),
            network.recurrent_weight.values(),
            float_data(workspace.d_hidden_previous),
            batch_size,
            hidden,
            gate_size,
            1.0f
        );
        std::swap(workspace.d_hidden, workspace.d_hidden_previous);
        std::swap(workspace.d_cell, workspace.d_cell_previous);
    }
}

}  // namespace
}  // namespace gameagent
