#include "RecurrentPolicyBackend.h"
#include "StreamProtocol.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path checkpoint_path() {
    if (const char* value = std::getenv("GAMEAGENT_RESNET50_CHECKPOINT");
        value != nullptr && value[0] != '\0') {
        return value;
    }
    return "/workspace/resnet50_best.ckpt";
}

gameagent::RecurrentPolicyBackendConfig make_config(
    const std::filesystem::path& directory,
    bool resume
) {
    gameagent::RecurrentPolicyBackendConfig config;
    config.backend = "cpp";
    config.checkpoint_path =
        directory / "shared_recurrent_latest.baseai.ckpt";
    config.resnet50_checkpoint_path = checkpoint_path();
    config.action_sizes = {3};
    config.visual_size = 2048;
    config.projection_size = 8;
    config.controller_size = 2;
    config.temporal_hidden_size = 8;
    config.context_size = 3;
    config.strategy_hidden_size = 8;
    config.sequence_capacity = 2;
    config.character_embedding_size = 2;
    config.opponent_embedding_size = 2;
    config.opponent_count = 1;
    config.replay_capacity = 8;
    config.minimum_replay_size = 2;
    config.batch_size = 2;
    config.training_updates_per_transition = 1;
    config.target_update_interval = 100;
    config.n_step_options = 1;
    config.learning_rate = 1.0e-4f;
    config.gradient_clip = 5.0f;
    config.terminal_replay_fraction = 0.0f;
    config.hard_negative_replay_fraction = 0.0f;
    config.winning_replay_fraction = 0.0f;
    config.winning_replay_min_fraction = 0.0f;
    config.trainable_resnet = true;
    config.resnet_learning_rate = 1.0e-6f;
    config.resnet_batch_size = 2;
    config.resnet_update_interval = 1;
    config.resnet_replay_images = 8;
    config.training_enabled = true;
    config.checkpoint_writes_enabled = true;
    config.resume = resume;
    return config;
}

}  // namespace

int main() {
    try {
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
            device_count == 0) {
            std::cout << "[SKIP] CUDA device unavailable\n";
            return 77;
        }
        if (!std::filesystem::is_regular_file(checkpoint_path())) {
            std::cout << "[SKIP] ImageNet checkpoint unavailable: "
                      << checkpoint_path() << '\n';
            return 77;
        }
        constexpr std::string_view encoded =
            "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/2wBDAQMDAwQDBAgEBAgQCwkLEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBD/wAARCAACAAIDAREAAhEBAxEB/8QAFAABAAAAAAAAAAAAAAAAAAAAB//EABQQAQAAAAAAAAAAAAAAAAAAAAD/xAAUAQEAAAAAAAAAAAAAAAAAAAAH/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/aAAwDAQACEQMRAD8AIi2L3//Z";
        const std::vector<std::uint8_t> jpeg =
            gameagent::decode_base64(encoded);
        const std::filesystem::path directory =
            std::filesystem::temp_directory_path() /
            ("gameagent_cpp_stage4_" + std::to_string(getpid()));
        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);

        double updated_checksum = 0.0;
        std::uint64_t updated_count = 0;
        {
            auto config = make_config(directory, false);
            gameagent::RecurrentPolicyBackend backend(config);
            const auto initial = backend.stats();
            require(initial.trainable_resnet,
                    "native backend did not enable stage4");
            require(initial.resnet_checkpoint_loaded,
                    "ImageNet checkpoint was not loaded");
            require(initial.resnet_training_updates == 0,
                    "fresh stage4 unexpectedly has optimizer updates");
            const std::array<float, 2> controller{0.0f, 0.0f};
            const std::array<float, 3> context{0.1f, -0.2f, 0.3f};
            const std::array<std::uint8_t, 3> mask{1, 1, 1};
            for (std::size_t sample = 0; sample < 2; ++sample) {
                const std::uint64_t frame = 10U + sample * 2U;
                backend.observe(
                    {}, jpeg, frame, controller, context, true
                );
                const auto q = backend.q_values(0, 0, 3);
                backend.begin_transition(1, 0, 0, q[1]);
                backend.observe(
                    {}, jpeg, frame + 1U, controller, context, false
                );
                backend.close_transition(
                    1, 1.0f, true, false, mask
                );
            }
            updated_checksum = backend.stats().resnet_parameter_checksum;
            updated_count = backend.stats().resnet_training_updates;
            require(updated_count == 1,
                    "stage4 did not perform exactly one update");
            require(std::isfinite(updated_checksum) &&
                    updated_checksum != initial.resnet_parameter_checksum,
                    "stage4 weights did not change");
            backend.save(config.checkpoint_path, false);
        }

        const std::filesystem::path stage4_checkpoint =
            directory / "resnet50_stage4_latest.baseai.ckpt";
        require(std::filesystem::is_regular_file(stage4_checkpoint),
                "stage4 checkpoint was not saved");
        {
            auto config = make_config(directory, true);
            gameagent::RecurrentPolicyBackend resumed(config);
            require(resumed.stats().resnet_checkpoint_loaded,
                    "resumed backend did not validate ImageNet initialization");
            require(resumed.stats().resnet_training_updates == updated_count,
                    "stage4 AdamW step was not restored");
            require(std::abs(
                        resumed.stats().resnet_parameter_checksum -
                        updated_checksum
                    ) < 1.0e-9,
                    "stage4 weights were not restored exactly");
        }
        std::filesystem::remove_all(directory);
        std::cout << "[INFO] resnet_updates=" << updated_count
                  << " checksum=" << updated_checksum << '\n'
                  << "[PASS] Native C++/CUDA ResNet50 stage4 train/save/resume passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
