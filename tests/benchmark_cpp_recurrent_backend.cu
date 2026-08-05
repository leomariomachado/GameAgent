#include "RecurrentPolicyBackend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void observe(
    gameagent::RecurrentPolicyBackend& backend,
    std::uint64_t frame_id,
    bool episode_started
) {
    std::vector<float> visual(2048);
    std::vector<float> controller(12);
    std::vector<float> context(30);
    for (std::size_t index = 0; index < visual.size(); ++index) {
        visual[index] = std::sin(
            static_cast<float>(frame_id) * 0.011f +
            static_cast<float>(index) * 0.007f
        );
    }
    controller[frame_id % controller.size()] = 1.0f;
    for (std::size_t index = 0; index < context.size(); ++index) {
        context[index] = static_cast<float>(
            (frame_id + index) % 101
        ) / 100.0f;
    }
    backend.observe(
        visual, {}, frame_id, controller, context, episode_started
    );
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
        gameagent::RecurrentPolicyBackendConfig config;
        config.backend = "cpp";
        config.checkpoint_path =
            "/tmp/gameagent_cpp_recurrent_benchmark.baseai.ckpt";
        config.action_sizes = {32, 32, 32, 32};
        config.replay_capacity = 128;
        config.minimum_replay_size = 64;
        config.batch_size = 64;
        config.n_step_options = 1;
        config.training_updates_per_transition = 1;
        config.target_update_interval = 500;
        config.training_enabled = true;
        config.checkpoint_writes_enabled = false;
        config.resume = false;
        config.trainable_resnet = false;
        gameagent::RecurrentPolicyBackend backend(config);

        std::uint64_t frame_id = 1;
        for (std::size_t sample = 0; sample < 64; ++sample) {
            for (std::size_t step = 0;
                 step < config.sequence_capacity;
                 ++step) {
                observe(backend, frame_id++, step == 0);
            }
            const std::vector<float> q = backend.q_values(0, 0, 32);
            backend.begin_transition(0, 0, 0, q[0]);
            observe(backend, frame_id++, false);
            const std::vector<std::uint8_t> mask(32, 1);
            const auto begin = Clock::now();
            const auto result = backend.close_transition(
                1, 1.0f, true, true, mask
            );
            const auto end = Clock::now();
            if (sample == 63) {
                if (!result.training_performed) {
                    throw std::runtime_error(
                        "production-size training update was not performed"
                    );
                }
                std::cout << "[BENCHMARK] train_update_ms="
                          << milliseconds(begin, end) << '\n';
            }
        }

        for (std::size_t step = 0;
             step < config.sequence_capacity;
             ++step) {
            observe(backend, frame_id++, step == 0);
        }
        static_cast<void>(backend.q_values(0, 0, 32));
        constexpr int kIterations = 100;
        const auto inference_begin = Clock::now();
        for (int iteration = 0; iteration < kIterations; ++iteration) {
            static_cast<void>(backend.q_values(0, 0, 32));
        }
        const auto inference_end = Clock::now();
        std::cout
            << "[BENCHMARK] inference_ms="
            << milliseconds(inference_begin, inference_end) / kIterations
            << " updates=" << backend.stats().training_updates
            << " parameters_checksum="
            << backend.stats().parameter_checksum << '\n'
            << "[PASS] Production-size native recurrent benchmark passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
