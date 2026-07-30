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
      temporal_lstm_(make_lstm_config(config_)) {
    if (config_.resnet50_checkpoint.empty()) {
        throw std::invalid_argument(
            "temporal agent requires a ResNet50 checkpoint"
        );
    }
    if (config_.external_visual_encoder &&
        config_.frozen_lstm_inference_enabled) {
        throw std::invalid_argument(
            "external visual encoder cannot use the frozen CUDA LSTM"
        );
    }
    if (!config_.external_visual_encoder) {
        feature_extractor_ = std::make_unique<ResNet50FeatureExtractor>(
            config_.device_id,
            config_.verify_resnet50_checkpoint
        );
        checkpoint_report_ = feature_extractor_->load_checkpoint(
            config_.resnet50_checkpoint.string()
        );
    } else if (!std::filesystem::is_regular_file(
                   config_.resnet50_checkpoint
               )) {
        throw std::invalid_argument(
            "external ResNet50 checkpoint was not found"
        );
    }
    if (!config_.temporal_policy_checkpoint.empty()) {
        temporal_checkpoint_report_ = temporal_lstm_.load_checkpoint(
            config_.temporal_policy_checkpoint.string()
        );
    }

    // Warm up CUDA, the frozen backbone and the temporal kernels before a
    // gameplay connection is accepted. The synthetic step is discarded.
    if (feature_extractor_ != nullptr) {
        base::data::image::Image warmup_image(
            ResNet50FeatureExtractor::kInputWidth,
            ResNet50FeatureExtractor::kInputHeight,
            ResNet50FeatureExtractor::kInputChannels
        );
        std::fill(
            warmup_image.bytes().begin(), warmup_image.bytes().end(), 0U
        );
        const __half* feature =
            feature_extractor_->extract_device(warmup_image);
        temporal_lstm_.push(
            feature,
            {},
            false,
            0.0f,
            {
                .frame_id = 1,
                .frame_timestamp_ms = 1,
                .decision_id = 1,
                .decision_timestamp_ms = 1,
                .applied_decision_id = 0,
                .applied_command_known = false,
            },
            feature_extractor_->stream()
        );
        if (config_.frozen_lstm_inference_enabled) {
            static_cast<void>(
                temporal_lstm_.infer(feature_extractor_->stream())
            );
        }
        temporal_lstm_.reset(feature_extractor_->stream());
    }
}

TemporalObservationResult TemporalAgent::observe_frame(
    const BufferedFrame& frame
) {
    if (frame.frame_id == 0 || frame.received_timestamp_ms <= 0 ||
        frame.image.empty()) {
        throw std::invalid_argument(
            "temporal observation contains an invalid decoded frame"
        );
    }

    std::lock_guard lock(mutex_);
    const auto started = std::chrono::steady_clock::now();
    if (config_.external_visual_encoder) {
        if (frame.jpeg.empty()) {
            throw std::invalid_argument(
                "trainable ResNet50 requires the original JPEG"
            );
        }
        latest_visual_jpeg_ = frame.jpeg;
        latest_visual_feature_ = nullptr;
        latest_visual_feature_host_.clear();
    } else {
        latest_visual_feature_ =
            feature_extractor_->extract_device(frame.image);
        latest_visual_feature_host_ =
            feature_extractor_->download_device_feature(
                latest_visual_feature_
            );
        latest_visual_jpeg_.clear();
    }
    latest_frame_metadata_ = {
        .frame_id = frame.frame_id,
        .frame_timestamp_ms =
            frame.client_timestamp_ms > 0
                ? frame.client_timestamp_ms
                : frame.received_timestamp_ms,
    };
    const auto finished = std::chrono::steady_clock::now();
    return {
        .frame_id = frame.frame_id,
        .sequence_size =
            config_.external_visual_encoder
                ? external_steps_.size()
                : temporal_lstm_.size(),
        .feature_ms =
            std::chrono::duration<double, std::milli>(
                finished - started
            ).count(),
    };
}

TemporalAgentOutput TemporalAgent::advance_and_infer(
    const std::optional<AppliedActionRecord>& latest_applied_action,
    std::uint64_t decision_id,
    std::int64_t decision_timestamp_ms
) {
    if (decision_id == 0 || decision_timestamp_ms <= 0) {
        throw std::invalid_argument(
            "temporal decision tick has invalid metadata"
        );
    }
    std::lock_guard lock(mutex_);
    if ((!config_.external_visual_encoder &&
         latest_visual_feature_ == nullptr) ||
        (config_.external_visual_encoder &&
         latest_visual_jpeg_.empty()) ||
        latest_frame_metadata_.frame_id == 0) {
        throw std::runtime_error(
            "temporal decision requires a cached visual feature"
        );
    }

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
            decision_timestamp_ms -
                latest_applied_action
                    ->acknowledgement_received_timestamp_ms
        );
        command_age_seconds = std::min(
            1.0f, static_cast<float>(age_ms) / 1000.0f
        );
    }

    TemporalStepMetadata metadata = latest_frame_metadata_;
    metadata.decision_id = decision_id;
    metadata.decision_timestamp_ms = decision_timestamp_ms;
    metadata.applied_decision_id = applied_decision_id;
    metadata.applied_command_known = command_known;
    if (feature_extractor_ != nullptr) {
        temporal_lstm_.push(
            latest_visual_feature_,
            command_state,
            command_known,
            command_age_seconds,
            metadata,
            feature_extractor_->stream()
        );
    } else {
        if (external_steps_.size() == config_.sequence_capacity) {
            external_steps_.pop_front();
        }
        external_steps_.push_back(metadata);
    }

    TemporalAgentOutput output;
    if (config_.frozen_lstm_inference_enabled) {
        output.lstm =
            temporal_lstm_.infer(feature_extractor_->stream());
    } else {
        output.lstm.hidden.assign(config_.hidden_size, 0.0f);
        output.lstm.steps =
            config_.external_visual_encoder
                ? std::vector<TemporalStepMetadata>(
                      external_steps_.begin(),
                      external_steps_.end()
                  )
                : temporal_lstm_.steps();
    }
    output.visual_feature = latest_visual_feature_host_;
    output.visual_jpeg = latest_visual_jpeg_;
    output.visual_frame_id = latest_frame_metadata_.frame_id;
    std::copy(
        command_state.begin(),
        command_state.end(),
        output.controller_features.begin()
    );
    output.controller_features[10] = command_known ? 1.0f : 0.0f;
    output.controller_features[11] = std::clamp(
        command_age_seconds, 0.0f, 1.0f
    );
    if (
        config_.frozen_lstm_inference_enabled &&
        temporal_lstm_.policy_loaded()
    ) {
        output.action_prediction =
            temporal_lstm_.predict_from_hidden(
                feature_extractor_->stream()
            );
    }
    double squared_norm = 0.0;
    for (float value : output.lstm.hidden) {
        squared_norm += static_cast<double>(value) * value;
    }
    output.hidden_l2 = static_cast<float>(std::sqrt(squared_norm));
    return output;
}

TemporalAgentOutput TemporalAgent::infer() {
    std::lock_guard lock(mutex_);
    TemporalAgentOutput output;
    if (config_.frozen_lstm_inference_enabled) {
        output.lstm =
            temporal_lstm_.infer(feature_extractor_->stream());
    } else {
        output.lstm.hidden.assign(config_.hidden_size, 0.0f);
        output.lstm.steps =
            config_.external_visual_encoder
                ? std::vector<TemporalStepMetadata>(
                      external_steps_.begin(),
                      external_steps_.end()
                  )
                : temporal_lstm_.steps();
    }
    output.visual_feature = latest_visual_feature_host_;
    output.visual_jpeg = latest_visual_jpeg_;
    output.visual_frame_id = latest_frame_metadata_.frame_id;
    if (
        config_.frozen_lstm_inference_enabled &&
        temporal_lstm_.policy_loaded()
    ) {
        output.action_prediction =
            temporal_lstm_.predict_from_hidden(
                feature_extractor_->stream()
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
    temporal_lstm_.reset(
        feature_extractor_ != nullptr
            ? feature_extractor_->stream()
            : nullptr
    );
    latest_visual_feature_ = nullptr;
    latest_visual_feature_host_.clear();
    latest_visual_jpeg_.clear();
    external_steps_.clear();
    latest_frame_metadata_ = {};
}

bool TemporalAgent::ready() const {
    std::lock_guard lock(mutex_);
    return config_.external_visual_encoder
        ? !latest_visual_jpeg_.empty()
        : latest_visual_feature_ != nullptr;
}

std::size_t TemporalAgent::sequence_size() const {
    std::lock_guard lock(mutex_);
    return config_.external_visual_encoder
        ? external_steps_.size()
        : temporal_lstm_.size();
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
