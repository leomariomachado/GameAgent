#pragma once

#include "ActionHistory.h"
#include "FrameBuffer.h"
#include "ResNet50FeatureExtractor.h"
#include "TemporalLstm.h"

#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <array>
#include <vector>

namespace gameagent {

struct TemporalAgentConfig {
    std::filesystem::path resnet50_checkpoint;
    std::filesystem::path temporal_policy_checkpoint;
    std::size_t sequence_capacity = 8;
    std::size_t visual_projection_size = 256;
    std::size_t hidden_size = 256;
    int device_id = 0;
    std::uint32_t initialization_seed = 0x4c53544dU;
    bool verify_resnet50_checkpoint = true;
    bool frozen_lstm_inference_enabled = true;
    // When enabled, the recurrent backend owns the ResNet50 forward/backward
    // graph. The temporal bridge forwards the original in-memory JPEG and
    // does not allocate another frozen backbone.
    bool external_visual_encoder = false;
};

struct TemporalObservationResult {
    std::uint64_t frame_id = 0;
    std::size_t sequence_size = 0;
    double feature_ms = 0.0;
};

struct TemporalAgentOutput {
    TemporalLstmOutput lstm;
    std::optional<TemporalActionPrediction> action_prediction;
    // Visual observation for the newest camera frame and controller state.
    // With an external trainable encoder the JPEG is forwarded in memory;
    // otherwise this contains the frozen 2048-value feature.
    std::vector<float> visual_feature;
    std::vector<std::uint8_t> visual_jpeg;
    std::uint64_t visual_frame_id = 0;
    std::array<
        float,
        TemporalLstmConfig::kControllerFeatureCount
    > controller_features{};
    float hidden_l2 = 0.0f;
};

// End-to-end temporal feature path:
//
// decoded RGB frame -> ResNet50 -> cached/forwarded visual observation
//                   + latest client-confirmed command -> GPU LSTM
//
// A loaded policy may predict active/release plus controller state for
// diagnostics. Without one, every character uses the same deterministic
// from-scratch temporal encoder and learns only through its own DQN.
class TemporalAgent {
public:
    explicit TemporalAgent(TemporalAgentConfig config);

    // Updates only the cached visual feature. The recurrent sequence advances
    // in advance_and_infer(), once per controller decision, even when several
    // decisions reuse the same camera frame.
    TemporalObservationResult observe_frame(const BufferedFrame& frame);
    TemporalAgentOutput advance_and_infer(
        const std::optional<AppliedActionRecord>& latest_applied_action,
        std::uint64_t decision_id,
        std::int64_t decision_timestamp_ms
    );
    TemporalAgentOutput infer();
    void reset();

    [[nodiscard]] bool ready() const;
    [[nodiscard]] std::size_t sequence_size() const;
    [[nodiscard]] const ResNet50CheckpointReport& checkpoint_report() const;
    [[nodiscard]] const std::optional<TemporalPolicyCheckpointReport>&
    temporal_checkpoint_report() const;
    [[nodiscard]] const TemporalAgentConfig& config() const;

private:
    TemporalAgentConfig config_;
    std::unique_ptr<ResNet50FeatureExtractor> feature_extractor_;
    TemporalLstm temporal_lstm_;
    ResNet50CheckpointReport checkpoint_report_;
    std::optional<TemporalPolicyCheckpointReport>
        temporal_checkpoint_report_;
    const __half* latest_visual_feature_ = nullptr;
    std::vector<float> latest_visual_feature_host_;
    std::vector<std::uint8_t> latest_visual_jpeg_;
    std::deque<TemporalStepMetadata> external_steps_;
    TemporalStepMetadata latest_frame_metadata_;
    mutable std::mutex mutex_;
};

}  // namespace gameagent
