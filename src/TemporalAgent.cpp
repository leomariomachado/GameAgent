#include "TemporalAgent.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace gameagent {
namespace {

TemporalLstmConfig make_lstm_config(const TemporalAgentConfig& config) {
    return {
        .visual_size =
            static_cast<std::size_t>(ResNet50FeatureExtractor::kFeatureSize),
        .visual_projection_size = config.visual_projection_size,
        .hidden_size = config.hidden_size,
        .sequence_capacity = config.sequence_capacity,
        .device_id = config.device_id,
        .initialization_seed = config.initialization_seed,
    };
}

}  // namespace

TemporalAgent::TemporalAgent(TemporalAgentConfig config)
    : config_(std::move(config)),
      feature_extractor_(
          config_.device_id,
          config_.verify_resnet50_checkpoint
      ),
      temporal_lstm_(make_lstm_config(config_)) {
    if (config_.resnet50_checkpoint.empty()) {
        throw std::invalid_argument(
            "temporal agent requires a ResNet50 checkpoint"
        );
    }
    checkpoint_report_ = feature_extractor_.load_checkpoint(
        config_.resnet50_checkpoint.string()
    );
    if (!config_.temporal_policy_checkpoint.empty()) {
        temporal_checkpoint_report_ = temporal_lstm_.load_checkpoint(
            config_.temporal_policy_checkpoint.string()
        );
    }

    // Warm up CUDA, the frozen backbone and the temporal kernels before a
    // gameplay connection is accepted. The synthetic step is discarded.
    base::data::image::Image warmup_image(
        ResNet50FeatureExtractor::kInputWidth,
        ResNet50FeatureExtractor::kInputHeight,
        ResNet50FeatureExtractor::kInputChannels
    );
    std::fill(
        warmup_image.bytes().begin(), warmup_image.bytes().end(), 0U
    );
    const __half* feature =
        feature_extractor_.extract_device(warmup_image);
    temporal_lstm_.push(
        feature,
        {},
        false,
        0.0f,
        {
            .frame_id = 1,
            .frame_timestamp_ms = 1,
            .applied_decision_id = 0,
            .applied_command_known = false,
        },
        feature_extractor_.stream()
    );
    static_cast<void>(
        temporal_lstm_.infer(feature_extractor_.stream())
    );
    temporal_lstm_.reset(feature_extractor_.stream());
}

TemporalObservationResult TemporalAgent::observe(
    const BufferedFrame& frame,
    const std::optional<AppliedActionRecord>& latest_applied_action
) {
    if (frame.frame_id == 0 || frame.received_timestamp_ms <= 0 ||
        frame.image.empty()) {
        throw std::invalid_argument(
            "temporal observation contains an invalid decoded frame"
        );
    }

    std::lock_guard lock(mutex_);
    const auto started = std::chrono::steady_clock::now();
    const __half* feature =
        feature_extractor_.extract_device(frame.image);

    std::array<float, 10> command_state{};
    bool command_known = false;
    std::uint64_t applied_decision_id = 0;
    float command_age_seconds = 0.0f;
    if (latest_applied_action.has_value()) {
        command_state = latest_applied_action->command_state;
        command_known = true;
        applied_decision_id = latest_applied_action->decision_id;
        const std::int64_t age_ms = std::max<std::int64_t>(
            0,
            frame.received_timestamp_ms -
                latest_applied_action
                    ->acknowledgement_received_timestamp_ms
        );
        command_age_seconds = std::min(
            1.0f, static_cast<float>(age_ms) / 1000.0f
        );
    }

    temporal_lstm_.push(
        feature,
        command_state,
        command_known,
        command_age_seconds,
        {
            .frame_id = frame.frame_id,
            .frame_timestamp_ms =
                frame.client_timestamp_ms > 0
                    ? frame.client_timestamp_ms
                    : frame.received_timestamp_ms,
            .applied_decision_id = applied_decision_id,
            .applied_command_known = command_known,
        },
        feature_extractor_.stream()
    );
    const auto finished = std::chrono::steady_clock::now();
    return {
        .frame_id = frame.frame_id,
        .applied_decision_id = applied_decision_id,
        .applied_command_known = command_known,
        .sequence_size = temporal_lstm_.size(),
        .enqueue_ms =
            std::chrono::duration<double, std::milli>(
                finished - started
            ).count(),
    };
}

TemporalAgentOutput TemporalAgent::infer() {
    std::lock_guard lock(mutex_);
    TemporalAgentOutput output{
        .lstm = temporal_lstm_.infer(feature_extractor_.stream()),
    };
    if (temporal_lstm_.policy_loaded()) {
        output.action_prediction =
            temporal_lstm_.predict_from_hidden(
                feature_extractor_.stream()
            );
    }
    double squared_norm = 0.0;
    for (float value : output.lstm.hidden) {
        squared_norm += static_cast<double>(value) * value;
    }
    output.hidden_l2 = static_cast<float>(std::sqrt(squared_norm));
    return output;
}

void TemporalAgent::reset() {
    std::lock_guard lock(mutex_);
    temporal_lstm_.reset(feature_extractor_.stream());
}

bool TemporalAgent::ready() const {
    std::lock_guard lock(mutex_);
    return temporal_lstm_.ready();
}

std::size_t TemporalAgent::sequence_size() const {
    std::lock_guard lock(mutex_);
    return temporal_lstm_.size();
}

const ResNet50CheckpointReport& TemporalAgent::checkpoint_report() const {
    return checkpoint_report_;
}

const std::optional<TemporalPolicyCheckpointReport>&
TemporalAgent::temporal_checkpoint_report() const {
    return temporal_checkpoint_report_;
}

const TemporalAgentConfig& TemporalAgent::config() const {
    return config_;
}

}  // namespace gameagent
