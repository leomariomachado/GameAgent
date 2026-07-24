#include "ActionHistory.h"
#include "TemporalAgent.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string environment_or(const char* name, std::string fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0'
        ? value
        : std::move(fallback);
}

gameagent::BufferedFrame make_frame(
    std::uint64_t frame_id,
    std::int64_t timestamp_ms
) {
    base::data::image::Image image(512, 448, 3);
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            image.at(y, x, 0) = static_cast<std::uint8_t>(
                (x + frame_id * 3U) % 256U
            );
            image.at(y, x, 1) = static_cast<std::uint8_t>(
                (y + frame_id * 5U) % 256U
            );
            image.at(y, x, 2) = static_cast<std::uint8_t>(
                (x + y + frame_id * 7U) % 256U
            );
        }
    }
    return {
        .frame_id = frame_id,
        .client_timestamp_ms = timestamp_ms,
        .received_timestamp_ms = timestamp_ms + 5,
        .jpeg_size = 0,
        .image = std::move(image),
    };
}

float maximum_absolute_difference(
    const std::vector<float>& left,
    const std::vector<float>& right
) {
    require(left.size() == right.size(), "hidden-state sizes differ");
    float maximum = 0.0f;
    for (std::size_t index = 0; index < left.size(); ++index) {
        maximum = std::max(
            maximum, std::abs(left[index] - right[index])
        );
    }
    return maximum;
}

void require_finite_nonzero(const gameagent::TemporalAgentOutput& output) {
    require(!output.lstm.hidden.empty(), "temporal hidden state is empty");
    require(
        std::all_of(
            output.lstm.hidden.begin(),
            output.lstm.hidden.end(),
            [](float value) { return std::isfinite(value); }
        ),
        "temporal hidden state contains a non-finite value"
    );
    require(
        std::isfinite(output.hidden_l2) && output.hidden_l2 > 0.0f,
        "temporal hidden state has invalid norm"
    );
}

}  // namespace

int main() {
    try {
        const std::filesystem::path checkpoint = environment_or(
            "GAMEAGENT_RESNET50_CHECKPOINT",
            "/workspace/resnet50_best.ckpt"
        );
        if (!std::filesystem::is_regular_file(checkpoint)) {
            std::cout << "[SKIP] ResNet50 checkpoint not found: "
                      << checkpoint << '\n';
            return 77;
        }
        const int device_id = std::stoi(
            environment_or("GAMEAGENT_DEVICE_ID", "0")
        );
        const std::filesystem::path temporal_checkpoint = environment_or(
            "GAMEAGENT_TEMPORAL_CHECKPOINT", ""
        );
        const bool has_temporal_checkpoint =
            !temporal_checkpoint.empty() &&
            std::filesystem::is_regular_file(temporal_checkpoint);

        gameagent::TemporalAgent agent({
            .resnet50_checkpoint = checkpoint,
            .temporal_policy_checkpoint =
                has_temporal_checkpoint
                    ? temporal_checkpoint
                    : std::filesystem::path{},
            .sequence_capacity = 8,
            .visual_projection_size = 256,
            .hidden_size = 256,
            .device_id = device_id,
            .initialization_seed = 12345,
            .verify_resnet50_checkpoint = true,
        });
        require(
            agent.checkpoint_report().tensor_count == 320,
            "temporal agent did not load the complete ResNet50 checkpoint"
        );
        require(!agent.ready(), "warm-up observation leaked into temporal state");

        gameagent::ActionHistory action_history(16);
        action_history.record_sent(1, 100, {3}, 1'000);
        const auto acknowledged_a = action_history.acknowledge(
            {
                .decision_id = 1,
                .applied_timestamp_ms = 1'010,
                .applied_commands = {3},
            },
            1'015
        );
        require(acknowledged_a.accepted, "test A command ACK was rejected");
        auto latest = action_history.latest_applied(1);
        require(latest.size() == 1, "confirmed command was not retained");

        const auto frame = make_frame(1, 1'020);
        const auto observed_a = agent.observe(frame, latest.back());
        require(
            observed_a.applied_command_known &&
                observed_a.applied_decision_id == 1,
            "temporal observation did not use the confirmed A command"
        );
        const auto output_a = agent.infer();
        require_finite_nonzero(output_a);
        if (has_temporal_checkpoint) {
            require(
                agent.temporal_checkpoint_report().has_value() &&
                    output_a.action_prediction.has_value() &&
                    output_a.action_prediction->probabilities.size() == 10,
                "trained temporal action head was not loaded"
            );
        }
        require(
            output_a.lstm.steps.size() == 1 &&
                output_a.lstm.steps.back().frame_id == 1 &&
                output_a.lstm.steps.back().applied_decision_id == 1,
            "temporal metadata is not aligned with its ResNet50 feature"
        );

        agent.reset();
        action_history.record_sent(2, 101, {9}, 1'030);
        const auto acknowledged_right = action_history.acknowledge(
            {
                .decision_id = 2,
                .applied_timestamp_ms = 1'040,
                .applied_commands = {9},
            },
            1'045
        );
        require(
            acknowledged_right.accepted,
            "test Right command ACK was rejected"
        );
        latest = action_history.latest_applied(1);
        agent.observe(frame, latest.back());
        const auto output_right = agent.infer();
        require_finite_nonzero(output_right);
        require(
            maximum_absolute_difference(
                output_a.lstm.hidden, output_right.lstm.hidden
            ) > 1.0e-6f,
            "changing the confirmed command did not change the LSTM state"
        );

        agent.reset();
        std::vector<float> first_sequence_output;
        for (std::uint64_t frame_id = 1; frame_id <= 9; ++frame_id) {
            agent.observe(
                make_frame(
                    frame_id,
                    2'000 + static_cast<std::int64_t>(frame_id) * 67
                ),
                latest.back()
            );
        }
        const auto wrapped = agent.infer();
        require_finite_nonzero(wrapped);
        require(
            wrapped.lstm.steps.size() == 8,
            "temporal ring did not enforce its configured capacity"
        );
        require(
            wrapped.lstm.steps.front().frame_id == 2 &&
                wrapped.lstm.steps.back().frame_id == 9,
            "temporal ring did not retain the eight newest ResNet50 features"
        );
        first_sequence_output = wrapped.lstm.hidden;

        agent.reset();
        for (std::uint64_t frame_id = 1; frame_id <= 9; ++frame_id) {
            agent.observe(
                make_frame(
                    frame_id,
                    2'000 + static_cast<std::int64_t>(frame_id) * 67
                ),
                latest.back()
            );
        }
        const auto repeated = agent.infer();
        require(
            maximum_absolute_difference(
                first_sequence_output, repeated.lstm.hidden
            ) <= 1.0e-6f,
            "temporal inference is not deterministic after reset"
        );

        std::cout
            << "[INFO] sequence=" << repeated.lstm.steps.size()
            << " hidden=" << repeated.lstm.hidden.size()
            << " hidden_l2=" << repeated.hidden_l2
            << " lstm_ms=" << repeated.lstm.inference_ms
            << " newest_frame=" << repeated.lstm.steps.back().frame_id
            << " applied_decision="
            << repeated.lstm.steps.back().applied_decision_id
            << '\n';
        if (has_temporal_checkpoint) {
            const auto& report = *agent.temporal_checkpoint_report();
            const auto& prediction = *repeated.action_prediction;
            std::cout
                << "[POLICY] epoch=" << report.epoch
                << " step=" << report.training_step
                << " best_val_loss=" << report.best_validation_loss
                << " parameters=" << report.parameter_count
                << " predicted_commands=";
            for (std::size_t index = 0;
                 index < prediction.commands.size();
                 ++index) {
                std::cout << (index == 0 ? "[" : ",")
                          << prediction.commands[index];
            }
            std::cout
                << (prediction.commands.empty() ? "[]" : "]")
                << '\n';
        }
        std::cout
            << "[PASS] ResNet50 + confirmed command + temporal LSTM test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
