#pragma once

#include "ActionHistory.h"
#include "FrameBuffer.h"
#include "ResNet50FeatureExtractor.h"
#include "TemporalLstm.h"

#include <cstddef>
#include <filesystem>
#include <mutex>
#include <optional>

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
};

struct TemporalObservationResult {
    std::uint64_t frame_id = 0;
    std::size_t sequence_size = 0;
    double feature_ms = 0.0;
};

struct TemporalAgentOutput {
    TemporalLstmOutput lstm;
    std::optional<TemporalActionPrediction> action_prediction;
    float hidden_l2 = 0.0f;
};

// End-to-end temporal feature path:
//
// decoded RGB frame -> frozen ResNet50 -> cached FP16[2048]
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
    ResNet50FeatureExtractor feature_extractor_;
    TemporalLstm temporal_lstm_;
    ResNet50CheckpointReport checkpoint_report_;
    std::optional<TemporalPolicyCheckpointReport>
        temporal_checkpoint_report_;
    const __half* latest_visual_feature_ = nullptr;
    TemporalStepMetadata latest_frame_metadata_;
    mutable std::mutex mutex_;
};

}  // namespace gameagent
