#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gameagent {

struct TemporalLstmConfig {
    static constexpr std::size_t kControllerCommandCount = 10;
    static constexpr std::size_t kControllerFeatureCount =
        kControllerCommandCount + 2;

    std::size_t visual_size = 2048;
    std::size_t visual_projection_size = 256;
    std::size_t hidden_size = 256;
    std::size_t sequence_capacity = 8;
    int device_id = 0;
    std::uint32_t initialization_seed = 0x4c53544dU;
};

struct TemporalStepMetadata {
    std::uint64_t frame_id = 0;
    std::int64_t frame_timestamp_ms = 0;
    std::uint64_t applied_decision_id = 0;
    bool applied_command_known = false;
};

struct TemporalLstmOutput {
    std::vector<float> hidden;
    std::vector<TemporalStepMetadata> steps;
    double inference_ms = 0.0;
};

struct TemporalPolicyCheckpointReport {
    std::uint32_t version = 0;
    std::uint64_t training_step = 0;
    std::uint64_t epoch = 0;
    float best_validation_loss = 0.0f;
    std::size_t parameter_count = 0;
};

struct TemporalActionPrediction {
    std::array<float, 10> probabilities{};
    std::vector<int> commands;
};

// Small inference-only GPU LSTM kept inside GameAgent while the temporal
// architecture is validated. Each step consumes:
//
//   projected ResNet50 feature
//   + 10-value confirmed controller state
//   + acknowledgement-known flag
//   + normalized age of the confirmed state
//
// The ResNet50 feature is copied device-to-device into a bounded ring, so a
// frame is never sent through the backbone again when later decisions reuse it.
class TemporalLstm {
public:
    explicit TemporalLstm(TemporalLstmConfig config);
    ~TemporalLstm();

    TemporalLstm(const TemporalLstm&) = delete;
    TemporalLstm& operator=(const TemporalLstm&) = delete;
    TemporalLstm(TemporalLstm&&) = delete;
    TemporalLstm& operator=(TemporalLstm&&) = delete;

    void push(
        const __half* device_visual_feature,
        const std::array<float, 10>& applied_command_state,
        bool applied_command_known,
        float applied_command_age_seconds,
        TemporalStepMetadata metadata,
        cudaStream_t stream
    );

    // Launches the complete sliding-window LSTM and returns its FP32 hidden
    // vector on the GPU. A future action head should consume this pointer on
    // the same stream without a host round-trip. The pointer is owned here.
    const float* infer_device(cudaStream_t stream);

    // Diagnostic convenience path used by tests and server metrics.
    TemporalLstmOutput infer(cudaStream_t stream);
    TemporalPolicyCheckpointReport load_checkpoint(
        const std::string& path
    );
    TemporalActionPrediction predict_from_hidden(
        cudaStream_t stream,
        float threshold = 0.6f
    );
    void reset(cudaStream_t stream);

    [[nodiscard]] bool ready() const;
    [[nodiscard]] bool policy_loaded() const;
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t capacity() const;
    [[nodiscard]] std::size_t hidden_size() const;
    [[nodiscard]] const TemporalLstmConfig& config() const;

private:
    void release() noexcept;
    void launch_forward(cudaStream_t stream);

    TemporalLstmConfig config_;
    std::size_t next_slot_ = 0;
    std::size_t size_ = 0;
    std::vector<TemporalStepMetadata> metadata_;

    __half* device_visual_ring_ = nullptr;
    float* device_controller_ring_ = nullptr;
    float* device_projection_weights_ = nullptr;
    float* device_projection_bias_ = nullptr;
    float* device_input_weights_ = nullptr;
    float* device_recurrent_weights_ = nullptr;
    float* device_gate_bias_ = nullptr;
    float* device_projection_ = nullptr;
    float* device_gates_ = nullptr;
    float* device_hidden_ = nullptr;
    float* device_cell_ = nullptr;
    float* device_action_head_weights_ = nullptr;
    float* device_action_head_bias_ = nullptr;
    float* device_action_logits_ = nullptr;
    bool policy_loaded_ = false;
};

}  // namespace gameagent
