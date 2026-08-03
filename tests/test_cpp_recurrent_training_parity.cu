#include "RecurrentPolicyBackend.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename T>
void read_values(std::istream& input, std::vector<T>& values) {
    for (T& value : values) {
        if (!(input >> value)) {
            throw std::runtime_error("truncated recurrent training fixture");
        }
    }
}

struct Step {
    std::vector<float> visual;
    std::vector<float> controller;
    std::vector<float> context;
};

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            throw std::invalid_argument(
                "usage: test INITIAL_CHECKPOINT FIXTURE OUTPUT_CHECKPOINT"
            );
        }
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
            device_count == 0) {
            std::cout << "[SKIP] CUDA device unavailable\n";
            return 77;
        }
        std::ifstream input(argv[2]);
        if (!input) {
            throw std::runtime_error("could not open training fixture");
        }

        gameagent::RecurrentPolicyBackendConfig config;
        std::size_t character_count = 0;
        input >> config.visual_size
              >> config.projection_size
              >> config.controller_size
              >> config.temporal_hidden_size
              >> config.context_size
              >> config.strategy_hidden_size
              >> config.sequence_capacity
              >> config.character_embedding_size
              >> config.opponent_embedding_size
              >> config.opponent_count
              >> character_count;
        config.action_sizes.resize(character_count);
        read_values(input, config.action_sizes);
        std::size_t character = 0;
        std::size_t opponent = 0;
        std::size_t step_count = 0;
        input >> character >> opponent >> step_count;
        if (step_count != config.sequence_capacity) {
            throw std::runtime_error("fixture sequence capacity mismatch");
        }

        std::vector<Step> steps(step_count);
        for (Step& step : steps) {
            step.visual.resize(config.visual_size);
            step.controller.resize(config.controller_size);
            step.context.resize(config.context_size);
            read_values(input, step.visual);
            read_values(input, step.controller);
            read_values(input, step.context);
        }
        std::vector<float> ignored_expected(config.action_sizes[character]);
        read_values(input, ignored_expected);
        Step next;
        next.visual.resize(config.visual_size);
        next.controller.resize(config.controller_size);
        next.context.resize(config.context_size);
        read_values(input, next.visual);
        read_values(input, next.controller);
        read_values(input, next.context);

        config.backend = "cpp";
        config.checkpoint_path = argv[1];
        config.replay_capacity = 16;
        config.minimum_replay_size = 4;
        config.batch_size = 4;
        config.training_updates_per_transition = 1;
        config.target_update_interval = 1000;
        config.gamma = 0.995f;
        config.n_step_options = 1;
        config.learning_rate = 1.0e-4f;
        config.gradient_clip = 5.0f;
        config.terminal_replay_fraction = 0.25f;
        config.hard_negative_replay_fraction = 0.25f;
        config.hard_negative_q_threshold = 1.0f;
        config.training_enabled = true;
        config.checkpoint_writes_enabled = true;
        config.resume = true;
        config.trainable_resnet = false;

        gameagent::RecurrentPolicyBackend backend(config);
        const std::uint64_t initial_updates =
            backend.stats().training_updates;
        for (std::size_t sample = 0; sample < 4; ++sample) {
            std::uint64_t frame_id = sample * 100 + 1;
            for (std::size_t index = 0; index < steps.size(); ++index) {
                const Step& step = steps[index];
                backend.observe(
                    step.visual, {}, frame_id++, step.controller,
                    step.context, index == 0
                );
            }
            const std::vector<float> q = backend.q_values(
                character, opponent, config.action_sizes[character]
            );
            backend.begin_transition(0, character, opponent, q[0]);
            backend.observe(
                next.visual, {}, frame_id, next.controller, next.context,
                false
            );
            const std::vector<std::uint8_t> mask(
                config.action_sizes[character], 1
            );
            backend.close_transition(1, 1.0f, true, mask);
        }
        if (backend.stats().training_updates != initial_updates + 1) {
            throw std::runtime_error(
                "native backend did not perform exactly one training update"
            );
        }
        backend.save(argv[3], false);
        std::cout
            << "[INFO] update=" << backend.stats().training_updates
            << " loss=" << backend.stats().last_loss << '\n'
            << "[PASS] Native one-step BPTT checkpoint generated\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
